"""Compile a SciFact-Open wiki store with a SPECIFIED model (for compiler-family
robustness, E1d). Same corpus/splits/pages as certify.py (seed 0); only the COMPILER
model changes. Saves store JSON in the same schema certify/cross_judge expect.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from certify import CLEAN_SYS, doc_text, split_claims, build_corpus, assign_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs", type=int, default=500)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--n_cal", type=int, default=40)
    ap.add_argument("--n_fpr", type=int, default=120)
    ap.add_argument("--k_rag", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--builder_port", type=int, default=8103)
    ap.add_argument("--builder_model", default="llama3.1-8b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "store_e1_llama.json"))
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, _ = split_claims(claims_all, corpus_all, args.n_cal, args.seed)
    corpus = build_corpus(corpus_all, cal_claims + audit_claims, args.n_docs, args.seed)
    pages = assign_pages(corpus.keys(), args.docs_per_page, args.seed)

    llm = VLLM(host=f"http://127.0.0.1:{args.builder_port}", model=args.builder_model)
    t0 = time.time()
    def one(members):
        blob = "\n\n".join(f"[Doc {k+1}] {doc_text(corpus[i])}" for k, i in enumerate(members))
        return llm.gen(CLEAN_SYS, blob, max_new_tokens=600, temperature=0.7)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        wiki_pages = list(ex.map(one, pages))
    cfg = {"n_docs": args.n_docs, "docs_per_page": args.docs_per_page, "n_cal": args.n_cal,
           "n_fpr": args.n_fpr, "k_rag": args.k_rag, "seed": args.seed,
           "builder_model": args.builder_model}
    json.dump({"pages": wiki_pages, "page_members": pages, "config": cfg},
              open(args.out, "w"))
    print(f"built {len(wiki_pages)} pages with {args.builder_model} in "
          f"{(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
