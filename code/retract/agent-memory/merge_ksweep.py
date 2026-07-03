"""Merge K-sweep parts -> results/k_sweep.json with per-round RSR curves + Wilson CIs."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["A1_auto", "A3_auto", "A4"]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


recs = []
K_MAX = 0
for f in sorted(glob.glob(f"{ROOT}/results/k_sweep_part_*.json")):
    d = json.load(open(f))
    K_MAX = max(K_MAX, d.get("K_MAX", 0))
    recs.extend(d.get("per_fact", []))
seen, uniq = set(), []
for r in recs:
    if r["id"] not in seen:
        seen.add(r["id"]); uniq.append(r)
recs = sorted(uniq, key=lambda r: r["id"])

curves = {}
for a in ARMS:
    per_round = []
    for k in range(K_MAX):
        ys = [r[a]["curve"][k] for r in recs
              if a in r and "curve" in r[a] and len(r[a]["curve"]) > k]
        n = len(ys); bf = sum(ys)
        per_round.append({"round": k + 1, "n": n,
                          "RSR": round(bf / n, 4) if n else None,
                          "wilson95": wilson(bf, n)})
    curves[a] = per_round
summ = {"n_facts": len(recs), "K_MAX": K_MAX, "curves": curves}
json.dump({"summary": summ, "per_fact": recs},
          open(f"{ROOT}/results/k_sweep.json", "w"), indent=2)
# compact print: RSR at each round
for a in ARMS:
    print(a, [c["RSR"] for c in curves[a]])
