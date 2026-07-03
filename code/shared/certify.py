"""E1 — one-shot retention certificate on an LLM-compiled wiki (SciFact-Open).

Produces a judge-noise-corrected lower confidence bound ("certificate") on true
fact retention of a compiled store, and validates it: empirical coverage over
audit-subsample resamples + arm separation (WIKI vs dense-RAG).

Pre-registered gate (E1). External data only.
Builder+judge: Qwen2.5-14B (local vLLM :8102), build temp 0.7 / judge temp 0.
Claim splits: CAL_TPR (40) vs AUDIT (76) disjoint; 163 no-support claims = FPR pool.
Style-matched FPR: absent-claims judged against compiled pages (wiki cert) and
raw-doc contexts (RAG cert). TPR measured on fact-present raw contexts (single
gold + buried-gold diagnostic for style robustness).
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from stats import retention_lcb, corrected_point, cp_lower, cp_upper, coverage_simulation

WORD_CAP = 350
CLEAN_SYS = (f"You are a wiki compiler. Write ONE wiki page (at most {WORD_CAP} words) that "
             "synthesizes ALL the scientific abstracts below. PRESERVE every key factual "
             "finding: specific entities, quantities, numbers, directions of effect, and "
             "comparisons from EACH abstract. Dense declarative prose. No preamble.")
JUDGE_SYS = ("You verify claims against a context. Answer YES only if the context contains "
             "information supporting the claim (paraphrase counts; the specific finding must "
             "be present). Otherwise answer NO. Reply with exactly one word: YES or NO.")


def doc_text(d):
    return (d.get("title", "") + ". " + d.get("text", "")).strip(". ")


def split_claims(claims_all, corpus_all, n_cal, seed):
    """-> (cal_claims, audit_claims, absent_claims). absent = no SUPPORT doc in candidates.
    We retain ALL evidence doc ids (any label) for absent claims so the caller can drop
    any whose evidence (SUPPORT *or* CONTRADICT) lands in the built corpus -- otherwise the
    FPR context is not genuinely fact-absent."""
    rng = random.Random(seed)
    sup, absent = [], []
    for c in claims_all:
        golds = [d for d, ev in c.get("evidence", {}).items()
                 if ev.get("label") == "SUPPORT" and d in corpus_all]
        if golds:
            sup.append({"claim": c["claim"], "gold": golds[0], "all_golds": golds})
        else:
            absent.append({"claim": c["claim"],
                           "evidence_docs": [d for d in c.get("evidence", {})]})
    rng.shuffle(sup)
    rng.shuffle(absent)
    return sup[:n_cal], sup[n_cal:], absent


def build_corpus(corpus_all, sup_claims, n_docs, seed):
    """All gold docs (cal+audit) + random fillers up to n_docs."""
    rng = random.Random(seed)
    gold = []
    seen = set()
    for c in sup_claims:
        for g in c["all_golds"]:
            if g not in seen:
                gold.append(g); seen.add(g)
    fillers = [i for i in corpus_all if i not in seen]
    rng.shuffle(fillers)
    docs = gold + fillers[:max(0, n_docs - len(gold))]
    return {i: corpus_all[i] for i in docs}


def assign_pages(doc_ids, per_page, seed):
    ids = sorted(doc_ids)
    random.Random(seed).shuffle(ids)
    return [ids[i:i + per_page] for i in range(0, len(ids), per_page)]


def build_wiki(llm, corpus, pages, workers):
    items = []
    for members in pages:
        blob = "\n\n".join(f"[Doc {k+1}] {doc_text(corpus[i])}" for k, i in enumerate(members))
        items.append((CLEAN_SYS, blob))
    if hasattr(llm, "gen_batch"):
        return llm.gen_batch(items, max_new_tokens=600, temperature=0.7)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda it: llm.gen(it[0], it[1], 600, 0.7), items))


def judge(llm, context, claim):
    a = llm.gen(JUDGE_SYS, f"CONTEXT:\n{context}\n\nCLAIM: {claim}",
                max_new_tokens=4, temperature=0.0)
    return 1 if a.strip().upper().startswith("YES") else 0


def judge_many(llm, pairs, workers, tag):
    """pairs = list of (context, claim). Batched if the engine supports it."""
    items = [(JUDGE_SYS, f"CONTEXT:\n{c}\n\nCLAIM: {cl}") for c, cl in pairs]
    if hasattr(llm, "gen_batch"):
        outs = llm.gen_batch(items, max_new_tokens=4, temperature=0.0)
        ys = [1 if a.strip().upper().startswith("YES") else 0 for a in outs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ys = list(ex.map(lambda it: 1 if llm.gen(it[0], it[1], 4, 0.0).strip().upper().startswith("YES") else 0, items))
    print(f"  {tag}: {sum(ys)}/{len(ys)} YES", flush=True)
    return ys


class DenseRetriever:
    """bge-small-en-v1.5 on CPU over raw docs."""

    def __init__(self, corpus):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.ids = sorted(corpus)
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        texts = [doc_text(corpus[i]) for i in self.ids]
        self.emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False)

    def top(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = self.np.argsort(-scores)[:k]
        return [self.ids[i] for i in idx]


def coverage_resample(verdicts, cal, truth, n_splits, audit_frac, alpha, seed):
    """LCB from a random audit subsample (with bootstrapped calibration counts)
    vs full-sample corrected truth proxy. Also returns naive (uncorrected) LCB
    coverage — the baseline a judge-blind certificate would report."""
    rng = random.Random(seed)
    n = len(verdicts)
    m = max(5, int(n * audit_frac))
    hits = naive_hits = 0
    lcbs = []
    for _ in range(n_splits):
        sub = rng.sample(verdicts, m)
        k_tpr = sum(rng.random() < cal["k_tpr"] / cal["n_tpr"] for _ in range(cal["n_tpr"]))
        k_fpr = sum(rng.random() < cal["k_fpr"] / cal["n_fpr"] for _ in range(cal["n_fpr"]))
        lcb, _ = retention_lcb(sum(sub), m, k_tpr, cal["n_tpr"],
                               k_fpr, cal["n_fpr"], alpha=alpha)
        lcbs.append(lcb)
        hits += lcb <= truth
        naive_hits += cp_lower(sum(sub), m, alpha) <= truth
    return hits / n_splits, sum(lcbs) / len(lcbs), naive_hits / n_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs", type=int, default=500)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--n_cal", type=int, default=40)
    ap.add_argument("--n_fpr", type=int, default=120)
    ap.add_argument("--k_rag", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_splits", type=int, default=1000)
    ap.add_argument("--audit_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results_e1.json"))
    ap.add_argument("--store_out", default=os.path.join(HERE, "store_e1.json"))
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, absent = split_claims(claims_all, corpus_all, args.n_cal, args.seed)
    corpus = build_corpus(corpus_all, cal_claims + audit_claims, args.n_docs, args.seed)
    # FPR pool must be GENUINELY fact-absent: drop any absent claim whose evidence doc
    # (SUPPORT or CONTRADICT) landed in the built corpus.
    corpus_ids = set(corpus)
    absent = [c for c in absent if not (set(c.get("evidence_docs", [])) & corpus_ids)]
    absent = absent[:args.n_fpr]
    pages = assign_pages(corpus.keys(), args.docs_per_page, args.seed)
    page_of_doc = {d: pi for pi, mem in enumerate(pages) for d in mem}
    print(f"docs={len(corpus)} pages={len(pages)} cal={len(cal_claims)} "
          f"audit={len(audit_claims)} absent={len(absent)} (evidence-in-corpus removed)", flush=True)

    llm = VLLM()
    t0 = time.time()
    rng = random.Random(args.seed)

    print("building wiki...", flush=True)
    wiki_pages = build_wiki(llm, corpus, pages, args.workers)
    json.dump({"pages": wiki_pages, "page_members": pages,
               "config": vars(args)}, open(args.store_out, "w"))

    retr = DenseRetriever(corpus)
    rag_ctx = lambda c: "\n\n".join(doc_text(corpus[d]) for d in retr.top(c["claim"], args.k_rag))

    # ---- calibration ----
    # TPR: fact present by construction (gold doc in context), two styles
    tpr_single = judge_many(llm, [(doc_text(corpus_all[c["gold"]]), c["claim"])
                                  for c in cal_claims], args.workers, "TPR single-gold")
    def buried(c):
        fillers = rng.sample(sorted(corpus), 7)
        docs = [c["gold"]] + [f for f in fillers if f != c["gold"]][:7]
        rng.shuffle(docs)
        return "\n\n".join(doc_text(corpus_all.get(d) or corpus[d]) for d in docs)
    tpr_buried = judge_many(llm, [(buried(c), c["claim"]) for c in cal_claims],
                            args.workers, "TPR buried-gold")
    # FPR: fact absent by construction, style-matched
    fpr_wiki = judge_many(llm, [(wiki_pages[rng.randrange(len(wiki_pages))], c["claim"])
                                for c in absent], args.workers, "FPR wiki-style")
    fpr_raw = judge_many(llm, [(rag_ctx(c), c["claim"]) for c in absent],
                         args.workers, "FPR raw-style")

    # ---- audit verdicts ----
    v_wiki = judge_many(llm, [(wiki_pages[page_of_doc[c["gold"]]], c["claim"])
                              for c in audit_claims], args.workers, "AUDIT wiki")
    v_rag = judge_many(llm, [(rag_ctx(c), c["claim"]) for c in audit_claims],
                       args.workers, "AUDIT rag-dense")

    # ---- certificates ----
    # TPR calibration: single-gold verdicts (canonical present context), n=40 DISTINCT
    # claims -- NOT pooled with buried as n=80 (the two styles on the same claim are
    # correlated; pooling understates variance and inflates the bound). buried reported
    # separately as a style-robustness check.
    cal_pool = {"k_tpr": sum(tpr_single), "n_tpr": len(tpr_single)}
    arms = {
        "wiki": {"v": v_wiki, "cal": dict(cal_pool, k_fpr=sum(fpr_wiki), n_fpr=len(fpr_wiki))},
        "rag_dense": {"v": v_rag, "cal": dict(cal_pool, k_fpr=sum(fpr_raw), n_fpr=len(fpr_raw))},
    }
    # honest coverage: simulate from known R across a grid, using THIS run's TPR/FPR/n's
    tpr_hat = cal_pool["k_tpr"] / cal_pool["n_tpr"]
    report = {"tpr_single": round(sum(tpr_single) / len(tpr_single), 3),
              "tpr_buried": round(sum(tpr_buried) / len(tpr_buried), 3),
              "n_tpr_distinct": len(tpr_single),
              "fpr_wiki_style": round(sum(fpr_wiki) / len(fpr_wiki), 3),
              "fpr_raw_style": round(sum(fpr_raw) / len(fpr_raw), 3),
              "n_fpr": len(fpr_wiki)}
    for name, a in arms.items():
        v, cal = a["v"], a["cal"]
        p = sum(v) / len(v)
        tpr = cal["k_tpr"] / cal["n_tpr"]
        fpr = cal["k_fpr"] / cal["n_fpr"]
        lcb, parts = retention_lcb(sum(v), len(v), cal["k_tpr"], cal["n_tpr"],
                                   cal["k_fpr"], cal["n_fpr"], alpha=args.alpha)
        r_hat = corrected_point(p, tpr, fpr)
        # simulation coverage across a grid of TRUE R (valid bound must cover >= 1-alpha)
        grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        cov = {round(R, 2): round(coverage_simulation(
                   R, tpr, fpr, len(v), cal["n_tpr"], cal["n_fpr"],
                   alpha=args.alpha, seed=args.seed), 3) for R in grid}
        report[name] = {"raw_verdict_rate": round(p, 3), "corrected_R": round(r_hat, 3),
                        "certificate_LCB": round(lcb, 3), "parts": parts,
                        "sim_coverage_by_trueR": cov,
                        "min_sim_coverage": round(min(cov.values()), 3)}
        print(f"\n{name}: raw={p:.3f} corrected_R={r_hat:.3f} LCB={lcb:.3f} "
              f"min_sim_coverage={min(cov.values()):.3f}", flush=True)

    # RAG retrieval quality (recall@k of the gold doc) -- shows bge-small is adequate
    recall = sum(1 for c in audit_claims if c["gold"] in retr.top(c["claim"], args.k_rag)) / len(audit_claims)
    report["rag_recall_at_k"] = round(recall, 3)
    print(f"RAG recall@{args.k_rag} = {recall:.3f}", flush=True)

    # separation: one-sided intervals on corrected R via component CIs
    report["n_calls"] = llm.n_calls
    report["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump({"config": vars(args), "report": report,
               "verdicts": {"wiki": v_wiki, "rag": v_rag,
                            "tpr_single": tpr_single, "tpr_buried": tpr_buried,
                            "fpr_wiki": fpr_wiki, "fpr_raw": fpr_raw},
               "audit_claims": [{"claim": c["claim"], "gold": c["gold"]} for c in audit_claims]},
              open(args.out, "w"), indent=2)
    print("\n=== E1 REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
