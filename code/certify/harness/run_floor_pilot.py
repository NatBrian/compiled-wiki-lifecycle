"""DECISIVE pilot: does the structural floor actually appear, and does COMPLETENESS beat it?

The earlier B2 runs refuted the headline because (1) COMPILE was top-k-over-cards (no
completeness) and (2) SciFact claims lexically overlap their gold abstracts, so BM25 RAG
never misses -> no floor to beat.

This pilot fixes both:
  - COMPILE-ALL reads EVERY card (true completeness), feasible because cards are ~10x smaller.
  - PARAPHRASED condition rewrites the gold abstract to kill lexical overlap, the exact
    semantic / grep-can't / RAG-can't regime the novelty targets.

Headline metric: false_absence = fraction of SUPPORTED claims wrongly called NEI.
Expected: CLEAN -> RAG fine. PARAPHRASED -> RAG false_absence jumps (retrieval miss),
COMPILE-ALL stays low (it never retrieves; it scans all N).

Backbone: local gemma4:12b via ollama on GPU 2 (OllamaLLM).
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import data as D
import methods as M


def build_cards(llm, corpus, cache_path):
    if cache_path and os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        cards = {int(k): v for k, v in cached.items()}
        if set(cards.keys()) >= set(corpus.keys()):
            print(f"  loaded {len(cards)} cached cards", flush=True)
            return {d: cards[d] for d in corpus}
    print(f"  COMPILE build: {len(corpus)} cards (O(N) once)...", flush=True)
    cards = M.compile_corpus(llm, corpus, batch_size=8)
    if cache_path:
        json.dump({str(k): v for k, v in cards.items()}, open(cache_path, "w"))
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../benchmarks/scifact-open")
    ap.add_argument("--corpus", default="corpus_candidates.jsonl")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--n_claims", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_rag", type=int, default=5)
    ap.add_argument("--out", default="../results/floor_pilot.json")
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, args.corpus))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    print(f"corpus_pool={len(corpus_all)} claims={len(claims_all)}", flush=True)

    # pick tested supported claims FIRST, then build a corpus of size N that keeps ONLY
    # their gold docs (+ random fill). Lets us run small/cheap N without dropping gold.
    sup_all = D.supported_claims(claims_all, set(corpus_all.keys()))
    sup = sup_all[: args.n_claims]
    corpus = D.mini_corpus_for_claims(corpus_all, sup, args.N, seed=args.seed)
    # re-attach in_corpus_evidence restricted to this corpus
    sup = D.supported_claims(sup, set(corpus.keys()))
    print(f"N={len(corpus)} tested_supported_claims={len(sup)}", flush=True)
    if not sup:
        print("no supported claims in subset; raise N"); return

    from llm import OllamaLLM
    llm = OllamaLLM()
    print(f"backend=ollama model={llm.model_name}", flush=True)

    # ---- base cards (clean) ----
    cards = build_cards(llm, corpus, os.path.join(os.path.dirname(args.out),
                                                  f"cards_N{args.N}_s{args.seed}.json"))

    # ---- gold docs of tested claims (these get paraphrased in PARA condition) ----
    gold_ids = set()
    for c in sup:
        gold_ids |= set(c["in_corpus_evidence"].keys())
    print(f"paraphrasing {len(gold_ids)} gold docs (kills lexical overlap)...", flush=True)
    para_text, para_card = {}, {}
    for gi, did in enumerate(gold_ids):
        src = corpus[did]["title"] + ". " + corpus[did]["text"]
        pt = M.paraphrase_text(llm, src)
        para_text[did] = pt
        pc, _ = llm.chat(M.CARD_SYS, pt[:1500], max_new_tokens=64)
        para_card[did] = pc.replace("\n", " ")
        print(f"  [{gi+1}/{len(gold_ids)}] doc {did} paraphrased", flush=True)

    # ---- two corpora / two card stores ----
    corpus_para = {d: dict(corpus[d]) for d in corpus}
    for did in gold_ids:
        corpus_para[did] = {"doc_id": did, "title": corpus[did]["title"],
                            "text": para_text[did]}
    cards_para = dict(cards)
    for did in gold_ids:
        cards_para[did] = para_card[did]

    conditions = {
        "clean": (corpus, cards),
        "paraphrased": (corpus_para, cards_para),
    }

    out = {"config": vars(args), "results": {}}
    for cond, (corp, crd) in conditions.items():
        retr = M.BM25(corp)
        rag_fa, comp_fa, rows = 0, 0, []
        for ci, c in enumerate(sup):
            claim = c["claim"]
            r = M.method_rag(llm, claim, corp, retr, k=args.k_rag)
            a = M.method_compile_all(llm, claim, crd)
            rag_miss = int(r["label"] == "NEI")
            comp_miss = int(a["label"] == "NEI")
            rag_fa += rag_miss
            comp_fa += comp_miss
            # did RAG even retrieve the gold doc?
            gold = set(c["in_corpus_evidence"].keys())
            retrieved_gold = bool(gold & set(r.get("retrieved", [])))
            rows.append({"claim_id": c["id"], "rag": r["label"], "compile_all": a["label"],
                         "rag_retrieved_gold": retrieved_gold,
                         "n_cards_scanned": a["n_cards_scanned"]})
            print(f"[{cond}][{ci+1}/{len(sup)}] rag={r['label']} "
                  f"compile_all={a['label']} gold_retrieved={retrieved_gold}", flush=True)
        nq = len(sup)
        out["results"][cond] = {
            "rag_false_absence": round(rag_fa / nq, 3),
            "compile_all_false_absence": round(comp_fa / nq, 3),
            "rag_gold_retrieval_rate": round(
                sum(x["rag_retrieved_gold"] for x in rows) / nq, 3),
            "n": nq, "rows": rows,
        }
        print(f"=== {cond}: RAG fa={out['results'][cond]['rag_false_absence']} "
              f"COMPILE-ALL fa={out['results'][cond]['compile_all_false_absence']} "
              f"(RAG gold-retrieval={out['results'][cond]['rag_gold_retrieval_rate']}) ===",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)
    print("\n================ HEADLINE ================", flush=True)
    for cond in conditions:
        rr = out["results"][cond]
        print(f"  {cond:12s} RAG_fa={rr['rag_false_absence']:.3f}  "
              f"COMPILE-ALL_fa={rr['compile_all_false_absence']:.3f}  "
              f"RAG_gold_retr={rr['rag_gold_retrieval_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
