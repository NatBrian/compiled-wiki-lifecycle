"""Merge E0b-AUTO chunk parts (results/e0b_auto_part_*.json) -> results/e0b_auto.json."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["A1_auto", "A3_auto", "A4"]
N_CONSOLID = int(os.environ.get("P7_N_CONSOLID", "2"))


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


recs = []
for f in sorted(glob.glob(f"{ROOT}/results/e0b_auto_part_*.json")):
    recs.extend(json.load(open(f)).get("per_fact", []))
seen, uniq = set(), []
for r in recs:
    if r["id"] not in seen:
        seen.add(r["id"]); uniq.append(r)
recs = sorted(uniq, key=lambda r: r["id"])

summ = {"n_facts": len(recs), "n_consolid": N_CONSOLID, "arms": {}}
for a in ARMS:
    xs = [r[a] for r in recs if a in r and "backflowed" in r[a]]
    n = len(xs); bf = sum(x["backflowed"] for x in xs)
    summ["arms"][a] = {"n": n, "backflowed": bf,
                       "RSR": round(bf / n, 4) if n else None, "wilson95": wilson(bf, n)}
a1 = summ["arms"]["A1_auto"]; a3 = summ["arms"]["A3_auto"]
summ["backflow_exists"] = bool(a1["RSR"] and a1["RSR"] >= 0.10)
summ["gate_fixes"] = bool(a1["RSR"] and a3["RSR"] is not None and a3["RSR"] <= a1["RSR"] - 0.10)
summ["GATE_AUTO"] = ("PASS_backflow_and_gate" if summ["backflow_exists"] and summ["gate_fixes"]
                     else "PASS_backflow_only" if summ["backflow_exists"] else "FAIL_no_backflow")
json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/e0b_auto.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
