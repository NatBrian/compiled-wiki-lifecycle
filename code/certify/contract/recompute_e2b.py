"""Recompute E2b (churn) certificates with the REFINED calibration (single-gold TPR
n=40, clean FPR) from the new E1 run, applied to E2b's stored per-batch verdict rates.
The judge calibration is a property of the (same Qwen) judge, so we reuse E1's. CPU only.
Writes results_e2b_fixed.json and prints the breach trajectory.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import retention_lcb, corrected_point

e1 = json.load(open(os.path.join(HERE, "results_e1.json")))
vv = e1["verdicts"]
k_tpr, n_tpr = sum(vv["tpr_single"]), len(vv["tpr_single"])      # single-gold, n=40
k_fpr, n_fpr = sum(vv["fpr_wiki"]), len(vv["fpr_wiki"])          # clean, genuinely absent
tpr, fpr = k_tpr / n_tpr, k_fpr / n_fpr
print(f"refined cal: TPR {k_tpr}/{n_tpr}={tpr:.3f}  FPR {k_fpr}/{n_fpr}={fpr:.3f}")

e2b = json.load(open(os.path.join(HERE, "results_e2b.json")))
tl = e2b["timeline"]

def lcb_from(rawn):
    raw, n = rawn["raw"], rawn["n"]
    k = round(raw * n)
    lcb, _ = retention_lcb(k, n, k_tpr, n_tpr, k_fpr, n_fpr)
    return round(lcb, 4)

# stale cert = t1's stale (issued at t0); recompute once
stale_lcb = lcb_from(tl[1]["stale"])
rows = []
for e in tl[1:]:
    raw_full = e["full"]["raw"]
    truth = corrected_point(raw_full, tpr, fpr)
    m_lcb = lcb_from(e["maintained"])
    rows.append({"t": e["t"], "truth": round(truth, 3), "stale_lcb": stale_lcb,
                 "stale_valid": bool(stale_lcb <= truth + 1e-9),
                 "maint_lcb": m_lcb, "maint_valid": bool(m_lcb <= truth + 1e-9)})
    print(f"t{e['t']}: truth={truth:.3f} stale_lcb={stale_lcb} valid={rows[-1]['stale_valid']} | "
          f"maint_lcb={m_lcb} valid={rows[-1]['maint_valid']}")

breach = [r["t"] for r in rows if not r["stale_valid"]]
print(f"\nSTALE breached at batches: {breach}")
print(f"MAINTAINED valid all batches: {all(r['maint_valid'] for r in rows)}")
json.dump({"refined_cal": {"tpr": tpr, "fpr": fpr, "n_tpr": n_tpr, "n_fpr": n_fpr},
           "rows": rows, "first_breach": breach[0] if breach else None,
           "e3_lottery_disagree": e2b["e3_lottery_transfer"]["per_probe_disagree"]},
          open(os.path.join(HERE, "results_e2b_fixed.json"), "w"), indent=2)
print("wrote results_e2b_fixed.json")
