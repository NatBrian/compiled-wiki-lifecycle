"""Merge B2b parts -> results/b2b_necessity.json + summary."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["G_none", "G_hash", "G_cone", "G_nli"]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


recs = []
for f in sorted(glob.glob(f"{ROOT}/results/b2b_part_*.json")):
    recs.extend(json.load(open(f)).get("per_fact", []))
seen, uniq = set(), []
for r in recs:
    if r["id"] not in seen:
        seen.add(r["id"]); uniq.append(r)
recs = sorted(uniq, key=lambda r: r["id"])

summ = {"n_facts": len(recs), "arms": {}}
for a in ARMS:
    xs = [r[a] for r in recs if a in r and "leaked" in r[a]]
    n = len(xs); lk = sum(x["leaked"] for x in xs); ben = sum(x["benign_retained"] for x in xs)
    summ["arms"][a] = {"n": n, "leak": round(lk / n, 4) if n else None,
                       "leak_wilson95": wilson(lk, n),
                       "benign_retained": round(ben / n, 4) if n else None}
h = summ["arms"]["G_hash"]["leak"]; nli = summ["arms"]["G_nli"]["leak"]; cone = summ["arms"]["G_cone"]["leak"]
summ["hash_minus_nli"] = round((h or 0) - (nli or 0), 4)
summ["hash_minus_cone"] = round((h or 0) - (cone or 0), 4)
summ["nli_necessary"] = bool(h and nli is not None and (h - nli) >= 0.10)
json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/b2b_necessity.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
