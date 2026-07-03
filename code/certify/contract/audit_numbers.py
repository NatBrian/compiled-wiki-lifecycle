"""Re-derive every number quoted in the paper from the results JSONs (post-audit-fix set).
Any MISMATCH means the paper prose disagrees with the data."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

def g(p): return json.load(open(os.path.join(HERE, p)))
checks = []
def chk(name, got, want, tol=0.005):
    ok = abs(got - want) <= tol
    checks.append((name, got, want, ok))

# E1 (offline, refined calibration)
e1 = g("results_e1.json")["report"]
chk("E1 wiki LCB", e1["wiki"]["certificate_LCB"], 0.371)
chk("E1 wiki R", e1["wiki"]["corrected_R"], 0.609)
chk("E1 wiki raw", e1["wiki"]["raw_verdict_rate"], 0.487)
chk("E1 rag LCB", e1["rag_dense"]["certificate_LCB"], 0.740)
chk("E1 wiki min_sim_coverage>=0.95", e1["wiki"]["min_sim_coverage"], 0.999, tol=0.05)
chk("E1 tpr_single", e1["tpr_single"], 0.800)
chk("E1 n_tpr_distinct", e1["n_tpr_distinct"], 40, tol=0.5)
chk("E1 fpr_wiki", e1["fpr_wiki_style"], 0.000)
chk("E1 n_fpr clean", e1["n_fpr"], 120, tol=0.5)
chk("E1 rag recall@5", e1["rag_recall_at_k"], 0.961, tol=0.01)

# E1c / E1d cross-family
e1c = g("results_e1c.json")["report"]
chk("E1c llama tpr", e1c["tpr"], 0.800)
chk("E1c llama fpr_raw", e1c["fpr_raw"], 0.050, tol=0.005)
chk("E1c wiki R", e1c["wiki"]["corrected_R"], 0.576)
chk("E1c wiki LCB", e1c["wiki"]["certificate_LCB"], 0.343)
chk("E1c rag LCB", e1c["rag_dense"]["certificate_LCB"], 0.715)
e1d = g("results_e1d.json")["report"]
chk("E1d wiki R", e1d["wiki"]["corrected_R"], 0.559)
chk("E1d wiki LCB", e1d["wiki"]["certificate_LCB"], 0.329)
chk("E1d rag LCB", e1d["rag_dense"]["certificate_LCB"], 0.740)

# E2 (calibration-independent: cost ratio) + E3 lottery
e2 = g("results_e2e3.json")
chk("E2 maint/full cost", e2["timeline"][-1]["cost"]["maintained"] / e2["timeline"][-1]["cost"]["full"], 0.405, tol=0.01)
chk("E3 lottery disagree", e2["e3_lottery_transfer"]["per_probe_disagree"], 0.2333)

# E2b (recomputed with refined calibration)
e2b = g("results_e2b_fixed.json")
chk("E2b first breach batch", e2b["first_breach"], 7, tol=0.5)
chk("E2b stale_lcb", e2b["rows"][0]["stale_lcb"], 0.386, tol=0.005)
chk("E2b t8 truth", e2b["rows"][-1]["truth"], 0.378)
chk("E2b maint valid all", sum(r["maint_valid"] for r in e2b["rows"]), 8, tol=0.5)
chk("E2b lottery (200doc)", e2b["e3_lottery_disagree"], 0.1711)

# E4 repair (refined calibration)
e4 = g("results_e4.json")
chk("E4 targeted adaptive", e4["targeted"]["adaptive_invalid_cert"]["lcb"], 0.760, tol=0.01)
chk("E4 targeted honest", e4["targeted"]["honest_cert"]["lcb"], 0.244, tol=0.01)
chk("E4 overclaim gap ~0.52", e4["targeted"]["overclaim_lcb_gap"], 0.515, tol=0.01)
chk("E4 union holdout", e4["union"]["holdout"]["lcb"], 0.389, tol=0.01)
chk("E4 initial holdout", e4["initial"]["holdout"]["lcb"], 0.244, tol=0.01)

# E5 currency (refined, held-out calibration)
e5 = g("results_e5.json")["timeline"]
chk("E5 t0 SER_ub", e5[0]["SER_UB"], 1.0)
chk("E5 t8 retention LCB", e5[8]["maintained"]["retention_LCB"], 0.842)
chk("E5 t8 SER_ub", e5[8]["maintained"]["SER_UB"], 0.208)
chk("E5 t8 cost maint", e5[8]["cost_maintained"], 504, tol=0.5)
chk("E5 t8 cost full", e5[8]["cost_full"], 1080, tol=0.5)

# sizing
sz = g("results_sizing.json")
chk("sizing rhat", sz["rhat"], 0.609)
chk("sizing n76 LCB", [r for r in sz["rows"] if r["n"] == 76][0]["mean_LCB"], 0.372, tol=0.02)

print("=== NUMBER AUDIT (post-fix) ===")
bad = 0
for name, got, want, ok in checks:
    if not ok: bad += 1
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {name}: got {got} want {want}")
print(f"\n{len(checks)-bad}/{len(checks)} pass" + ("" if not bad else f"  ({bad} MISMATCH)"))
