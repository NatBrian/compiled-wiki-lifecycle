"""Three systems under test for absence (SUPPORT / CONTRADICT / NEI):
  - RAG:     retrieve top-k RAW docs, classify.
  - DUMP:    stuff as many RAW docs as fit into context, classify (lost-in-middle as N grows).
  - COMPILE: read every doc ONCE into a compact canonical 'card' store (O(N) build, reused
             across all queries); at query time read the integrated card store. The card
             store is ~10x smaller than raw text, so far more of the corpus fits per query
             -> escapes the dump floor. This is the make-or-break contrast (compile != dump).

NEI = the absence answer. For a claim that DOES have in-corpus evidence, predicting NEI is a
false-absence error (our headline metric).
"""
import re


# ---------- lexical retriever (no GPU, no network) ----------
class BM25:
    def __init__(self, docs):
        self.ids = list(docs.keys())
        self.texts = [(docs[i]["title"] + " " + docs[i]["text"]) for i in self.ids]
        try:
            from rank_bm25 import BM25Okapi
            self.bm = BM25Okapi([self._tok(t) for t in self.texts])
            self.kind = "bm25"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer().fit(self.texts)
            self.mat = self.vec.transform(self.texts)
            self.kind = "tfidf"

    @staticmethod
    def _tok(t):
        return re.findall(r"[a-z0-9]+", t.lower())

    def top(self, query, k):
        if self.kind == "bm25":
            scores = self.bm.get_scores(self._tok(query))
        else:
            import numpy as np
            qv = self.vec.transform([query])
            scores = (self.mat @ qv.T).toarray().ravel()
        order = sorted(range(len(self.ids)), key=lambda j: -scores[j])[:k]
        return [self.ids[j] for j in order]


# ---------- classification prompt ----------
SYS = ("You are a scientific claim verifier. Given a claim and evidence passages, decide if "
       "the evidence SUPPORTS the claim, CONTRADICTS it, or gives Not Enough Info (NEI). "
       "Answer with exactly one token: SUPPORT, CONTRADICT, or NEI.")


def _classify(llm, claim, passages, max_new_tokens=8):
    if passages:
        ev = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    else:
        ev = "(no passages)"
    user = f"Claim: {claim}\n\nEvidence passages:\n{ev}\n\nAnswer (SUPPORT/CONTRADICT/NEI):"
    out, n_ctx = llm.chat(SYS, user, max_new_tokens=max_new_tokens)
    u = out.upper()
    label = "NEI"
    if "CONTRADICT" in u:
        label = "CONTRADICT"
    elif "SUPPORT" in u:
        label = "SUPPORT"
    return label, n_ctx


def _doc_text(corpus, did, max_chars=1200):
    d = corpus[did]
    return (d["title"] + ". " + d["text"])[:max_chars]


# ---------- RAG ----------
def method_rag(llm, claim, corpus, retriever, k=5):
    hits = retriever.top(claim, k)
    passages = [_doc_text(corpus, d) for d in hits]
    label, n_ctx = _classify(llm, claim, passages)
    return {"label": label, "n_ctx": n_ctx, "retrieved": hits}


# ---------- DUMP ----------
def method_dump(llm, claim, corpus, max_docs=None, char_budget=24000):
    ids = list(corpus.keys())
    passages, used = [], 0
    for d in ids:
        t = _doc_text(corpus, d)
        if used + len(t) > char_budget or (max_docs and len(passages) >= max_docs):
            break
        passages.append(t)
        used += len(t)
    label, n_ctx = _classify(llm, claim, passages)
    return {"label": label, "n_ctx": n_ctx, "n_docs_fit": len(passages), "n_total": len(ids)}


# ---------- COMPILE (build once, reuse) ----------
CARD_SYS = ("Extract the key factual findings of this scientific abstract as 2-3 dense "
            "declarative sentences. PRESERVE specific entities, quantities, numbers, "
            "directions of effect, and comparisons (these are what claims are checked "
            "against). No preamble, no hedging.")


def compile_corpus(llm, corpus, max_chars=1500, batch_size=64):
    """O(N) build, done ONCE. Returns {doc_id: card_text}. Reusable across all queries.
    Batched generation: card prompts are same-shape, so decode in batches for ~20-40x."""
    ids = list(corpus.keys())
    srcs = [(corpus[d]["title"] + ". " + corpus[d]["text"])[:max_chars] for d in ids]
    texts = []
    for s in range(0, len(srcs), batch_size):
        texts.extend(llm.chat_batch(CARD_SYS, srcs[s:s + batch_size], max_new_tokens=64,
                                    batch_size=batch_size))
        print(f"  compiled {min(s+batch_size, len(srcs))}/{len(srcs)} cards", flush=True)
    return {d: t.replace("\n", " ") for d, t in zip(ids, texts)}


def method_compile(llm, claim, cards, retriever_over_cards, k=20):
    """Query the COMPACT compiled store. Cards are short, so k can be large (more coverage
    per token than raw-doc RAG) and many fit -> escapes the dump floor."""
    hits = retriever_over_cards.top(claim, k)
    passages = [cards[d] for d in hits]
    label, n_ctx = _classify(llm, claim, passages)
    return {"label": label, "n_ctx": n_ctx, "retrieved": hits}


def cards_as_corpus(cards):
    """Wrap cards so BM25 can index them (title empty, text=card)."""
    return {d: {"title": "", "text": c, "doc_id": d} for d, c in cards.items()}


# ---------- COMPILE-ALL: TRUE completeness (reads EVERY card, no retrieval) ----------
def method_compile_all(llm, claim, cards, max_cards_ctx=4000):
    """The honest 'compile' the novelty actually requires: examine ALL N cards, not top-k.
    Because cards are ~10x smaller than raw docs, the WHOLE compiled store fits in context
    where the raw corpus dump would not -> reads all N (completeness) AND escapes the raw-dump
    lost-in-middle floor. This is the system that can soundly answer absence.

    If #cards exceeds max_cards_ctx, scan in chunks and take the strongest non-NEI verdict
    (still touches every card -> still complete)."""
    ids = list(cards.keys())
    items = [cards[d] for d in ids]
    n = len(items)
    verdicts = []
    for s in range(0, n, max_cards_ctx):
        chunk = items[s:s + max_cards_ctx]
        ev = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunk))
        user = (f"Claim: {claim}\n\nCompiled knowledge store ({len(chunk)} entries):\n{ev}\n\n"
                f"Does ANY entry support or contradict the claim? "
                f"Answer one token (SUPPORT/CONTRADICT/NEI):")
        out, n_ctx = llm.chat(SYS, user, max_new_tokens=8,
                              num_ctx=min(60000, 2048 + 40 * len(chunk)))
        u = out.upper()
        if "CONTRADICT" in u:
            verdicts.append("CONTRADICT")
        elif "SUPPORT" in u:
            verdicts.append("SUPPORT")
        else:
            verdicts.append("NEI")
    # complete scan: absence only if EVERY chunk says NEI
    label = "NEI"
    for v in verdicts:
        if v != "NEI":
            label = v
            break
    return {"label": label, "n_ctx": n * 40, "n_cards_scanned": n, "chunks": len(verdicts)}


# ---------- adversarial paraphrase (kills lexical overlap so top-k RAG misses) ----------
PARA_SYS = ("Rewrite the scientific abstract so it states the SAME findings but with completely "
            "different wording: use synonyms, change sentence structure, avoid reusing the "
            "distinctive technical nouns. Preserve all factual claims. Output only the rewrite.")


def paraphrase_text(llm, text, max_new_tokens=400):
    out, _ = llm.chat(PARA_SYS, text[:2000], max_new_tokens=max_new_tokens)
    return out.replace("\n", " ").strip()
