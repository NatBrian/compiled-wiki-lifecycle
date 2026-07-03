"""Merge B1 chunk parts (results/b1_part_*.json) into results/b1_ladder.json."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["A0", "A1", "A2", "A3", "A4"]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


TAG = os.environ.get("P7_TAG", "")
recs = []
for f in sorted(glob.glob(f"{ROOT}/results/b1{TAG}_part_*.json")):
    recs.extend(json.load(open(f)).get("per_fact", []))
# dedupe by id
seen, uniq = set(), []
for r in recs:
    if r["id"] not in seen:
        seen.add(r["id"]); uniq.append(r)
recs = sorted(uniq, key=lambda r: r["id"])

summ = {"n_facts": len(recs), "arms": {}}
for a in ARMS:
    xs = [r[a] for r in recs if a in r and "backflowed" in r[a]]
    n = len(xs); bf = sum(x["backflowed"] for x in xs); br = sum(x["benign_retained"] for x in xs)
    summ["arms"][a] = {"n": n, "RSR": round(bf / n, 4) if n else None,
                       "RSR_wilson95": wilson(bf, n),
                       "benign_retained": round(br / n, 4) if n else None}
a2 = summ["arms"]["A2"]["RSR"]; a3 = summ["arms"]["A3"]["RSR"]
summ["discrimination_A2_minus_A3"] = round((a2 or 0) - (a3 or 0), 4)
summ["membership_not_correctness"] = bool(a2 and a3 is not None and (a2 - a3) >= 0.10)
json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/b1_ladder{TAG}.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
