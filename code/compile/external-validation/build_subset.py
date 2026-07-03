"""Build a small, self-contained SciFact-Open subset for the Post-1 experiment.

Run LOCALLY (where benchmarks/scifact-open/data lives). Produces data/scifact_subset.json
so the Kaggle notebook is self-contained and never needs the 500k-doc corpus.

Subset = N tracked SUPPORT claims (each with its supporting abstract = a "probe doc")
       + a pool of filler abstracts (used as the incoming document stream that forces rewrites).
"""
import json, os, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root (this file lives at code/compile/external-validation/) ; data lives at benchmarks/scifact-open/data
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCI = os.path.join(ROOT, "benchmarks", "scifact-open", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tracked", type=int, default=75, help="tracked SUPPORT claims")
    ap.add_argument("--n_filler", type=int, default=1500, help="filler abstracts in the stream pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "scifact_subset.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    claims = [json.loads(l) for l in open(os.path.join(SCI, "claims.jsonl"))]
    # tracked = claims with a single SUPPORT gold doc (clean "is it still supported?" semantics)
    tracked_raw = []
    for c in claims:
        ev = c.get("evidence") or {}
        sup = [d for d, e in ev.items() if e.get("label") == "SUPPORT"]
        if len(sup) >= 1:
            tracked_raw.append((c["claim"], str(sup[0])))
    rng.shuffle(tracked_raw)
    tracked_raw = tracked_raw[: args.n_tracked]
    need_ids = {d for _, d in tracked_raw}

    # stream corpus once: grab needed abstracts + reservoir-sample fillers
    id2abs, fillers = {}, []
    seen = 0
    with open(os.path.join(SCI, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            did = str(d["doc_id"])
            ab = " ".join(d.get("abstract") or []).strip()
            if not ab:
                continue
            if did in need_ids:
                id2abs[did] = {"doc_id": did, "title": d.get("title", ""), "abstract": ab}
            # reservoir sample fillers (exclude probe docs)
            elif len(ab.split()) >= 60:
                seen += 1
                if len(fillers) < args.n_filler:
                    fillers.append({"doc_id": did, "title": d.get("title", ""), "abstract": ab})
                else:
                    j = rng.randint(0, seen - 1)
                    if j < args.n_filler:
                        fillers[j] = {"doc_id": did, "title": d.get("title", ""), "abstract": ab}

    tracked = [{"claim": cl, "doc_id": did, **id2abs[did]} for cl, did in tracked_raw if did in id2abs]
    out = {"tracked": tracked, "fillers": fillers,
           "meta": {"dataset": "SciFact-Open", "n_tracked": len(tracked), "n_filler": len(fillers)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}: tracked={len(tracked)} fillers={len(fillers)}")


if __name__ == "__main__":
    main()
