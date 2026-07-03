"""Scaled E1: significance + STEELMAN RAG.

Adds vs run_moat.py:
  - multiple seeds (different claim/corpus draws) -> mean +/- spread
  - RAG at several k (k=20 steelman: give top-k every chance before claiming the
    moat is structural, not just 'k too small')
  - scan_card as the headline scan arm (compiled store; cheaper + best in E1);
    scan_raw optional (expensive) via --scan_raw

Same metric: false_absence on supported claims (NEI = wrong). Same two regimes
(natural / adversarial-paraphrase). Resident gemma4:12b over HTTP, no new GPU proc.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import data as D
import methods as M
import scan as S
import adversarial as A
from client import HTTPLLM
from run_moat import pick, raw_store


def run_regime(llm, name, corpus, cards, sup, krags, do_scan_raw):
    retr = M.BM25(corpus)
    raw = raw_store(corpus) if do_scan_raw else None
    methods = [f"rag{k}" for k in krags] + ["scan_card"] + (["scan_raw"] if do_scan_raw else [])
    res = {m: [] for m in methods}
    for ci, c in enumerate(sup):
        claim = c["claim"]
        row = {}
        for k in krags:
            row[f"rag{k}"] = M.method_rag(llm, claim, corpus, retr, k=k)
        row["scan_card"] = S.method_scan(llm, claim, cards, window=30)
        if do_scan_raw:
            row["scan_raw"] = S.method_scan(llm, claim, raw, window=12)
        for m, x in row.items():
            x["false_absence"] = int(x["label"] == "NEI")
            res[m].append(x)
        print(f"  [{name} {ci+1}/{len(sup)}] " +
              " ".join(f"{m}={row[m]['label']}" for m in methods), flush=True)
    return res


def fa(rs):
    return sum(x["false_absence"] for x in rs) / len(rs)


def build_adversarial(llm, corpus, sup):
    adv = {i: dict(d) for i, d in corpus.items()}
    ob, oa = [], []
    for c in sup:
        for did in c["in_corpus_evidence"]:
            if did not in adv or adv[did].get("_para"):
                continue
            orig = adv[did]["title"] + ". " + adv[did]["text"]
            para = A.paraphrase_gold(llm, c["claim"], orig)
            ob.append(A.lexical_overlap(c["claim"], orig))
            oa.append(A.lexical_overlap(c["claim"], para))
            adv[did]["text"] = para; adv[did]["title"] = ""; adv[did]["_para"] = True
    return adv, (sum(ob)/max(1,len(ob)), sum(oa)/max(1,len(oa)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", default="corpus_candidates.jsonl")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--n_claims", type=int, default=60)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--krags", default="5,20")
    ap.add_argument("--scan_raw", action="store_true")
    ap.add_argument("--out", default="results_moat_scale.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    krags = [int(k) for k in args.krags.split(",")]
    corpus_all = D.load_corpus(os.path.join(args.data, args.corpus))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    print(f"corpus={len(corpus_all)} claims={len(claims_all)} seeds={seeds} krags={krags}", flush=True)

    llm = HTTPLLM()
    per_seed = []
    for seed in seeds:
        print(f"\n##### SEED {seed} #####", flush=True)
        corpus, sup, gold = pick(corpus_all, claims_all, args.N, args.n_claims, seed)
        print(f"N={len(corpus)} claims={len(sup)} gold={len(gold)}", flush=True)
        t = time.time(); cards = M.compile_corpus(llm, corpus)
        print(f"compiled {len(cards)} in {time.time()-t:.0f}s", flush=True)

        nat = run_regime(llm, f"nat{seed}", corpus, cards, sup, krags, args.scan_raw)
        adv_corpus, (ob, oa) = build_adversarial(llm, corpus, sup)
        changed = [i for i in adv_corpus if adv_corpus[i].get("_para")]
        adv_cards = dict(cards); adv_cards.update(M.compile_corpus(llm, {i: adv_corpus[i] for i in changed}))
        adv = run_regime(llm, f"adv{seed}", adv_corpus, adv_cards, sup, krags, args.scan_raw)

        sd = {"seed": seed, "n": len(sup), "overlap": [round(ob,3), round(oa,3)],
              "natural": {m: round(fa(nat[m]),3) for m in nat},
              "adversarial": {m: round(fa(adv[m]),3) for m in adv}}
        per_seed.append(sd)
        print(f"  seed {seed} nat={sd['natural']} adv={sd['adversarial']}", flush=True)

    # aggregate mean across seeds
    methods = list(per_seed[0]["natural"].keys())
    def agg(reg, m):
        vals = [s[reg][m] for s in per_seed]
        mean = sum(vals)/len(vals)
        return round(mean,3), round((max(vals)-min(vals)),3), vals
    summary = {"per_seed": per_seed, "methods": methods,
               "mean": {reg: {m: agg(reg, m)[0] for m in methods} for reg in ("natural","adversarial")},
               "spread": {reg: {m: agg(reg, m)[1] for m in methods} for reg in ("natural","adversarial")}}
    json.dump({"config": vars(args), "summary": summary}, open(args.out,"w"), indent=2)

    print("\n=== AGG false_absence (mean over seeds, lower=better) ===", flush=True)
    hdr = "  %-12s " + " ".join(f"{m:>10s}" for m in methods)
    print(hdr % "regime", flush=True)
    for reg in ("natural","adversarial"):
        print(("  %-12s " % reg) + " ".join(f"{summary['mean'][reg][m]:>10.3f}" for m in methods), flush=True)
    sc = "scan_card"
    for k in krags:
        moat = summary["mean"]["adversarial"][f"rag{k}"] - summary["mean"]["adversarial"][sc]
        print(f"\n  MOAT adv(rag{k} - scan_card) = {moat:+.3f}", flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
