"""Adversarial-paraphrase stress test = the moat isolator.

WHY: On natural SciFact, gold abstracts share vocabulary with the claim, so BM25
top-k already retrieves them and RAG looks fine. That hides our advantage. Real
corpora, though, often state a fact in DIFFERENT words than the query (implicit /
paraphrased evidence). There, lexical retrieval silently drops the gold doc, and a
top-k system reports "not found" -> a FALSE absence it cannot detect.

This module rewrites each gold abstract to preserve its factual finding while
removing surface lexical overlap with the claim. After rewriting:
  - grep / BM25 / RAG  -> gold ranked low, missed -> predicts NEI (false absence)
  - whole-store SCAN   -> reads every card, judges meaning -> still finds it

That contrast is the make-or-break. It is a STRESS test (disclosed as such), not a
natural-distribution number; its purpose is to show the failure mode that only a
complete semantic scan escapes.
"""
PARA_SYS = (
    "Rewrite the scientific abstract so it reports the SAME factual finding but uses "
    "different wording: swap technical terms for synonyms or paraphrases, change "
    "sentence structure, and AVOID reusing the distinctive content words of the given "
    "claim. Keep it truthful and self-contained. Output only the rewritten abstract."
)


def paraphrase_gold(llm, claim, abstract, max_new_tokens=320):
    user = f"CLAIM (avoid its wording): {claim}\n\nABSTRACT:\n{abstract}\n\nRewritten abstract:"
    out, _ = llm.chat(PARA_SYS, user, max_new_tokens=max_new_tokens)
    return out.strip() or abstract


def lexical_overlap(claim, text):
    """Jaccard over content tokens — a cheap check that paraphrase reduced overlap."""
    import re
    stop = set("the a an of to in is are was were and or for on with that this it as by "
               "be can may from at not no does do has have which whether any".split())
    tok = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop and len(w) > 2}
    cw, tw = tok(claim), tok(text)
    if not cw or not tw:
        return 0.0
    return len(cw & tw) / len(cw | tw)
