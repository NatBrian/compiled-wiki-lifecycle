"""E1c — cross-family judge robustness. Closes the single-model limitation.

Compiler stays Qwen2.5-14B (we reuse the SAVED E1 wiki pages, store_e1.json); only the
JUDGE changes to a different family (Llama-3.1-8B-Instruct). We recompute the same audit
claims, RAG contexts, and present/absent calibration deterministically (seed 0) and
re-judge everything with the Llama server, then recompute the certificate.

If the certificate's validity (FPR low, TPR-FPR>0.3) and the wiki-vs-RAG separation hold
under an independent judge family, the contract is not an artifact of self-judging.
External data only.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from certify import (JUDGE_SYS, doc_text, split_claims, build_corpus, assign_pages,
                     DenseRetriever, judge_many)
from stats import retention_lcb, corrected_point, cp_upper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--store", default=os.path.join(HERE, "store_e1.json"))
    ap.add_argument("--judge_port", type=int, default=8103)
    ap.add_argument("--judge_model", default="llama3.1-8b")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results_e1c.json"))
    args = ap.parse_args()

    store = json.load(open(args.store))
    cfg = store["config"]
    wiki_pages = store["pages"]
    pages = store["page_members"]
    page_of_doc = {d: pi for pi, mem in enumerate(pages) for d in mem}

    # reconstruct splits + corpus deterministically (same seed/args as E1)
    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, absent = split_claims(claims_all, corpus_all, cfg["n_cal"], cfg["seed"])
    corpus = build_corpus(corpus_all, cal_claims + audit_claims, cfg["n_docs"], cfg["seed"])
    corpus_ids = set(corpus)
    absent = [c for c in absent if not (set(c.get("evidence_docs", [])) & corpus_ids)][:cfg["n_fpr"]]
    rng = random.Random(cfg["seed"])

    # judge = different family
    llm = VLLM(host=f"http://127.0.0.1:{args.judge_port}", model=args.judge_model)
    retr = DenseRetriever(corpus)
    rag_ctx = lambda c: "\n\n".join(doc_text(corpus[d]) for d in retr.top(c["claim"], cfg["k_rag"]))
    t0 = time.time()

    # calibration (same constructions as E1)
    tpr_single = judge_many(llm, [(doc_text(corpus_all[c["gold"]]), c["claim"])
                                  for c in cal_claims], args.workers, "xTPR single")
    def buried(c):
        fillers = rng.sample(sorted(corpus), 7)
        docs = [c["gold"]] + [f for f in fillers if f != c["gold"]][:7]
        rng.shuffle(docs)
        return "\n\n".join(doc_text(corpus_all.get(d) or corpus[d]) for d in docs)
    tpr_buried = judge_many(llm, [(buried(c), c["claim"]) for c in cal_claims], args.workers, "xTPR buried")
    fpr_wiki = judge_many(llm, [(wiki_pages[rng.randrange(len(wiki_pages))], c["claim"])
                                for c in absent], args.workers, "xFPR wiki")
    fpr_raw = judge_many(llm, [(rag_ctx(c), c["claim"]) for c in absent], args.workers, "xFPR raw")
    v_wiki = judge_many(llm, [(wiki_pages[page_of_doc[c["gold"]]], c["claim"])
                              for c in audit_claims], args.workers, "xAUDIT wiki")
    v_rag = judge_many(llm, [(rag_ctx(c), c["claim"]) for c in audit_claims], args.workers, "xAUDIT rag")

    cal_pool = {"k_tpr": sum(tpr_single), "n_tpr": len(tpr_single)}  # n=40 distinct, single-gold
    rep = {"judge": args.judge_model,
           "tpr": round(cal_pool["k_tpr"] / cal_pool["n_tpr"], 3),
           "tpr_buried": round(sum(tpr_buried) / len(tpr_buried), 3),
           "fpr_wiki": round(sum(fpr_wiki) / len(fpr_wiki), 3),
           "fpr_raw": round(sum(fpr_raw) / len(fpr_raw), 3)}
    for name, v, fpr in [("wiki", v_wiki, fpr_wiki), ("rag_dense", v_rag, fpr_raw)]:
        lcb, parts = retention_lcb(sum(v), len(v), cal_pool["k_tpr"], cal_pool["n_tpr"],
                                   sum(fpr), len(fpr), alpha=args.alpha)
        r = corrected_point(sum(v) / len(v), cal_pool["k_tpr"] / cal_pool["n_tpr"], sum(fpr) / len(fpr))
        rep[name] = {"raw": round(sum(v) / len(v), 3), "corrected_R": round(r, 3),
                     "certificate_LCB": round(lcb, 3), "parts": parts}
        print(f"{name}: raw={rep[name]['raw']} R={rep[name]['corrected_R']} LCB={lcb:.3f}", flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    rep["n_calls"] = llm.n_calls
    json.dump({"report": rep}, open(args.out, "w"), indent=2)
    print("\n=== E1c CROSS-JUDGE ===", flush=True)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
