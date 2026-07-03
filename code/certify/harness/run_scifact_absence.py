"""Driver: compile vs dump vs RAG on SciFact-Open absence (NEI).

Headline metric (claim-level):
  false_absence = fraction of SUPPORTED claims (have in-corpus evidence) wrongly called NEI.
  A method is "right" on a supported claim iff it predicts SUPPORT or CONTRADICT (non-NEI).

Also report false_refusal proxy on true-NEI claims if provided.

Usage (CPU smoke):
  python run_scifact_absence.py --data ../benchmarks/scifact-open --N 200 --n_claims 10 \
      --methods rag,dump,compile --device cpu
Real run (GPU, once a card is safely free):
  ... --device cuda:IDX --model Qwen/Qwen2.5-14B-Instruct --N 12000
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import data as D
import methods as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to scifact-open data dir (has claims.jsonl, corpus.jsonl)")
    ap.add_argument("--corpus", default="corpus.jsonl", help="corpus file (use corpus_candidates.jsonl for fast smoke)")
    ap.add_argument("--smoke", action="store_true", help="mini-corpus around only the tested claims (cheap CPU)")
    ap.add_argument("--N", type=int, default=200, help="corpus subset size")
    ap.add_argument("--n_claims", type=int, default=10, help="how many supported claims to test")
    ap.add_argument("--methods", default="rag,dump,compile")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k_rag", type=int, default=5)
    ap.add_argument("--k_compile", type=int, default=20)
    ap.add_argument("--dump_char_budget", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_smoke.json")
    args = ap.parse_args()

    corpus_path = os.path.join(args.data, args.corpus)
    claims_path = os.path.join(args.data, "claims.jsonl")
    print(f"loading corpus {corpus_path}", flush=True)
    corpus_all = D.load_corpus(corpus_path)
    claims_all = D.load_claims(claims_path)
    print(f"corpus={len(corpus_all)} claims={len(claims_all)}", flush=True)

    if args.smoke:
        # pick tested claims first, then build a tiny corpus around only their gold docs
        sup = D.supported_claims(claims_all, set(corpus_all.keys()))[: args.n_claims]
        corpus = D.mini_corpus_for_claims(corpus_all, sup, args.N, seed=args.seed)
    else:
        corpus = D.subsample_corpus(corpus_all, claims_all, args.N, seed=args.seed)
        sup = D.supported_claims(claims_all, set(corpus.keys()))[: args.n_claims]
    print(f"subset N={len(corpus)} supported_claims_tested={len(sup)}", flush=True)
    if not sup:
        print("NO supported claims in subset — raise N.", flush=True); return

    from llm import LLM
    t0 = time.time()
    llm = LLM(model=args.model, device=args.device)
    print(f"model={llm.model_name} device={args.device} ctx_limit={llm.context_limit()} "
          f"loaded in {time.time()-t0:.0f}s", flush=True)

    want = args.methods.split(",")
    retr = M.BM25(corpus) if ("rag" in want) else None
    cards = card_retr = None
    if "compile" in want:
        print("COMPILE build (O(N) once)...", flush=True)
        tb = time.time()
        cards = M.compile_corpus(llm, corpus)
        card_retr = M.BM25(M.cards_as_corpus(cards))
        print(f"compiled {len(cards)} cards in {time.time()-tb:.0f}s", flush=True)

    results = {m: [] for m in want}
    for ci, c in enumerate(sup):
        claim = c["claim"]
        for m in want:
            if m == "rag":
                r = M.method_rag(llm, claim, corpus, retr, k=args.k_rag)
            elif m == "dump":
                r = M.method_dump(llm, claim, corpus, char_budget=args.dump_char_budget)
            elif m == "compile":
                r = M.method_compile(llm, claim, cards, card_retr, k=args.k_compile)
            else:
                continue
            r["false_absence"] = int(r["label"] == "NEI")  # supported claim -> NEI is wrong
            r["claim_id"] = c["id"]
            results[m].append(r)
        print(f"[{ci+1}/{len(sup)}] " + " ".join(
            f"{m}={results[m][-1]['label']}" for m in want), flush=True)

    summary = {}
    for m in want:
        rs = results[m]
        fa = sum(x["false_absence"] for x in rs) / len(rs)
        avg_ctx = sum(x["n_ctx"] for x in rs) / len(rs)
        summary[m] = {"false_absence_rate": round(fa, 3),
                      "avg_query_ctx_tokens": round(avg_ctx, 0),
                      "n": len(rs)}
    out = {"config": vars(args), "summary": summary, "per_claim": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY (lower false_absence = better) ===", flush=True)
    for m in want:
        print(f"  {m:8s}  false_absence={summary[m]['false_absence_rate']:.3f}  "
              f"avg_ctx_tok={summary[m]['avg_query_ctx_tokens']:.0f}", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
