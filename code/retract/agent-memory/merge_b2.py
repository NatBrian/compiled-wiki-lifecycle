"""Merge B2 chunk parts -> results/b2_ablation.json + summary."""
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
for f in sorted(glob.glob(f"{ROOT}/results/b2_part_*.json")):
    recs.extend(json.load(open(f)).get("per_fact", []))
seen, uniq = set(), []
for r in recs:
    if r["id"] not in seen:
        seen.add(r["id"]); uniq.append(r)
recs = sorted(uniq, key=lambda r: r["id"])

summ = {"n_facts": len(recs), "arms": {}}
for a in ARMS:
    xs = [r[a] for r in recs if a in r and "leaked_semantic" in r[a]]
    n = len(xs)
    sem = sum(x["leaked_semantic"] for x in xs)
    sub = sum(x["leaked_substring"] for x in xs)
    ben = sum(x["benign_retained"] for x in xs)
    summ["arms"][a] = {"n": n,
                       "leak_semantic": round(sem / n, 4) if n else None,
                       "leak_semantic_wilson95": wilson(sem, n),
                       "leak_substring": round(sub / n, 4) if n else None,
                       "benign_retained": round(ben / n, 4) if n else None}
h = summ["arms"]["G_hash"]["leak_semantic"]
cone = summ["arms"]["G_cone"]["leak_semantic"]
nli = summ["arms"]["G_nli"]["leak_semantic"]
summ["hash_minus_cone"] = round((h or 0) - (cone or 0), 4)
summ["hash_minus_nli"] = round((h or 0) - (nli or 0), 4)
summ["semantic_membership_necessary"] = bool(h and ((h - (cone or 0)) >= 0.10 or (h - (nli or 0)) >= 0.10))
json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/b2_ablation.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
