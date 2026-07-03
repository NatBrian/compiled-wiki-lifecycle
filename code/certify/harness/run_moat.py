"""DECISIVE make-or-break: does whole-store SCAN beat top-k RAG on ABSENCE,
specifically when evidence is paraphrased (lexical overlap removed)?

Two regimes, same supported claims (in-corpus evidence exists -> correct = non-NEI):
  NATURAL      : original abstracts. Expect rag ~= scan (BM25 finds gold via overlap).
  ADVERSARIAL  : gold abstracts paraphrased to kill claim overlap.
                 Expect rag false_absence JUMPS (misses gold), scan stays low.

Methods:
  rag   : BM25 top-k raw docs -> classify
  scan  : whole-store map-reduce over ALL compiled cards (the contribution)

Headline metric: false_absence = supported claim wrongly labeled NEI (lower better).
The MOAT shows iff  (rag_fa_adv - scan_fa_adv)  is large while natural gap is small.

Runs against the resident llama-server over HTTP (no new GPU process).
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import data as D
import methods as M
import scan as S
import adversarial as A
from client import HTTPLLM


def pick(corpus_all, claims_all, N, n_claims, seed):
    sup = D.supported_claims(claims_all, set(corpus_all.keys()))
    random.Random(seed).shuffle(sup)
    sup = sup[:n_claims]
    gold = set()
    for c in sup:
        gold |= {d for d in c["in_corpus_evidence"]}
    rng = random.Random(seed)
    pool = [i for i in corpus_all if i not in gold]
    fill = rng.sample(pool, min(max(0, N - len(gold)), len(pool)))
    ids = set(gold) | set(fill)
    corpus = {i: dict(corpus_all[i]) for i in ids}
    return corpus, sup, gold


def raw_store(corpus, max_chars=400):
    """Compact RAW snippet per doc (no LLM compression). Scanning this isolates the
    COVERAGE moat (read-all vs top-k) from card compression loss."""
    return {i: (corpus[i]["title"] + ". " + corpus[i]["text"])[:max_chars] for i in corpus}


def run_regime(llm, name, corpus, cards, sup, k_rag, raw_window=12, card_window=30):
    retr = M.BM25(corpus)
    raw = raw_store(corpus)
    res = {"rag": [], "scan_raw": [], "scan_card": []}
    for ci, c in enumerate(sup):
        claim = c["claim"]
        r = M.method_rag(llm, claim, corpus, retr, k=k_rag)          # top-k raw
        sr = S.method_scan(llm, claim, raw, window=raw_window)        # read-all raw (moat)
        sc = S.method_scan(llm, claim, cards, window=card_window)     # read-all cards (cost/soundness)
        for m, x in (("rag", r), ("scan_raw", sr), ("scan_card", sc)):
            x["false_absence"] = int(x["label"] == "NEI")
            x["claim_id"] = c["id"]
            res[m].append(x)
        print(f"  [{name} {ci+1}/{len(sup)}] rag={r['label']} scan_raw={sr['label']} "
              f"scan_card={sc['label']}", flush=True)
    return res


def summarize(res):
    out = {}
    for m, rs in res.items():
        fa = sum(x["false_absence"] for x in rs) / len(rs)
        ctx = sum(x["n_ctx"] for x in rs) / len(rs)
        out[m] = {"false_absence_rate": round(fa, 3),
                  "avg_query_ctx_tokens": round(ctx, 0), "n": len(rs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", default="corpus_candidates.jsonl")
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--n_claims", type=int, default=30)
    ap.add_argument("--k_rag", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--out", default="results_moat.json")
    args = ap.parse_args()

    random.seed(args.seed)
    corpus_all = D.load_corpus(os.path.join(args.data, args.corpus))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    print(f"corpus={len(corpus_all)} claims={len(claims_all)}", flush=True)

    corpus, sup, gold = pick(corpus_all, claims_all, args.N, args.n_claims, args.seed)
    print(f"N={len(corpus)} tested_claims={len(sup)} gold_docs={len(gold)}", flush=True)

    llm = HTTPLLM(max_workers=args.max_workers)

    # compile cards ONCE over natural corpus
    t = time.time()
    cards = M.compile_corpus(llm, corpus)
    print(f"compiled {len(cards)} cards in {time.time()-t:.0f}s", flush=True)

    print("== NATURAL ==", flush=True)
    nat = run_regime(llm, "nat", corpus, cards, sup, args.k_rag)

    # ADVERSARIAL: paraphrase each gold doc, rebuild its raw text + card
    print("== building adversarial (paraphrase gold) ==", flush=True)
    adv_corpus = {i: dict(d) for i, d in corpus.items()}
    overlaps_before, overlaps_after = [], []
    for c in sup:
        for did in c["in_corpus_evidence"]:
            if did not in adv_corpus or adv_corpus[did].get("_para"):
                continue
            orig = adv_corpus[did]["title"] + ". " + adv_corpus[did]["text"]
            para = A.paraphrase_gold(llm, c["claim"], orig)
            overlaps_before.append(A.lexical_overlap(c["claim"], orig))
            overlaps_after.append(A.lexical_overlap(c["claim"], para))
            adv_corpus[did]["text"] = para
            adv_corpus[did]["title"] = ""
            adv_corpus[did]["_para"] = True
    ob = sum(overlaps_before) / max(1, len(overlaps_before))
    oa = sum(overlaps_after) / max(1, len(overlaps_after))
    print(f"  gold lexical overlap with claim: {ob:.3f} -> {oa:.3f}", flush=True)

    # recompile ONLY changed gold cards; keep rest
    adv_cards = dict(cards)
    changed = [i for i in adv_corpus if adv_corpus[i].get("_para")]
    new = M.compile_corpus(llm, {i: adv_corpus[i] for i in changed})
    adv_cards.update(new)

    print("== ADVERSARIAL ==", flush=True)
    adv = run_regime(llm, "adv", adv_corpus, adv_cards, sup, args.k_rag)

    summary = {"natural": summarize(nat), "adversarial": summarize(adv),
               "overlap_before": round(ob, 3), "overlap_after": round(oa, 3),
               "N": len(corpus), "n_claims": len(sup), "llm_calls": llm.n_calls}
    out = {"config": vars(args), "summary": summary,
           "per_claim": {"natural": nat, "adversarial": adv}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY (false_absence, lower=better) ===", flush=True)
    for reg in ("natural", "adversarial"):
        s = summary[reg]
        print(f"  {reg:12s} rag={s['rag']['false_absence_rate']:.3f}  "
              f"scan_raw={s['scan_raw']['false_absence_rate']:.3f}  "
              f"scan_card={s['scan_card']['false_absence_rate']:.3f}", flush=True)
    a = summary["adversarial"]
    moat = a["rag"]["false_absence_rate"] - a["scan_raw"]["false_absence_rate"]
    print(f"\n  MOAT (adv rag_fa - scan_raw_fa) = {moat:+.3f}   "
          f"[overlap {summary['overlap_before']:.2f}->{summary['overlap_after']:.2f}]", flush=True)
    print(f"  card R(T) cost: scan_card adv_fa={a['scan_card']['false_absence_rate']:.3f} "
          f"(compression-loss axis)", flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
