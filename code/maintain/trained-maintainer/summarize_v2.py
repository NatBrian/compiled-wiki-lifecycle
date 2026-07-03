"""Trained-maintainer stage: single source of truth for this stage's numbers.

Reads results/*.json and prints (and dumps results/summary_v2.json):
  - v2 tournament: per-arm R0, R12 mean+/-sd (across seeds), incorp mean+/-sd, KM/fit half-life
  - trained vs anchored: mean diff, bootstrap 95% CI, Welch p  (is "beats" defensible?)
  - clean-rebuild reference: mean +/- sd (single CI'd value)
  - budget-trap: (cap x f x maintainer) R_train / R_held / gap, loose vs tight dose-response
  - currency: vanilla vs trained displacement/entrenchment mean+/-sd over seeds
  - scale: 1.5B vs 14B vs trained R12 (band)
"""
import glob, json, math, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")  # bundled repo: no nested code/, scripts live here
TPR_FPR = None


def _corr_RT(run):
    return run["timeline"][-1]["metrics"]["corrected"]


def _incorp(run):
    xs = [e["incorp_rate"] for e in run["timeline"] if e.get("incorp_rate") is not None]
    return float(np.mean(xs)) if xs else None


def load_arm(tag):
    fs = sorted(glob.glob(os.path.join(RES, f"results_v2_{tag}_seed*.json")))
    return [json.load(open(f)) for f in fs], fs


def boot_ci(a, b, n=10000, seed=0):
    """Bootstrap 95% CI for mean(a)-mean(b) (independent seeds)."""
    rng = np.random.RandomState(seed)
    a, b = np.array(a, float), np.array(b, float)
    diffs = [rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
             for _ in range(n)]
    return float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def welch_p(a, b):
    a, b = np.array(a, float), np.array(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return 0.0 if a.mean() != b.mean() else 1.0
    t = (a.mean() - b.mean()) / math.sqrt(va / na + vb / nb + 1e-12)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1 + 1e-9) + (vb / nb) ** 2 / (nb - 1 + 1e-9) + 1e-12)
    # two-sided p via survival of t (normal approx for small df is rough; use scipy if available)
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:
        from math import erfc
        return float(erfc(abs(t) / math.sqrt(2)))


out = {"tournament": {}, "clean_ref": None, "trained_vs_anchored": {},
       "trap": [], "currency": {}, "scale": {}}

print("=" * 70, "\nV2 TOURNAMENT (seeded sampling, track_incorp; mean +/- sd over seeds)\n", "=" * 70)
arms = ["vanilla", "conservative", "anchored", "trained", "proto"]
rt = {}
clean_vals = []
for tag in arms:
    runs, fs = load_arm(tag)
    if not runs:
        print(f"{tag:<14} (no v2 runs yet)"); continue
    RTs = [_corr_RT(r) for r in runs]
    R0s = [r["timeline"][0]["metrics"]["corrected"] for r in runs]
    incs = [_incorp(r) for r in runs if _incorp(r) is not None]
    rt[tag] = RTs
    # clean band at last checkpoint (std flavor)
    for r in runs:
        for cp in r.get("checkpoints", []):
            if cp["clean"].get("std", {}).get("corrected_mean") is not None and cp["t"] > 0:
                clean_vals.append(cp["clean"]["std"]["corrected_mean"])
    out["tournament"][tag] = {
        "n": len(RTs), "R0_mean": round(np.mean(R0s), 4),
        "R12_mean": round(np.mean(RTs), 4), "R12_sd": round(np.std(RTs), 4),
        "R12_per_seed": [round(x, 4) for x in RTs],
        "incorp_mean": round(np.mean(incs), 4) if incs else None,
        "incorp_sd": round(np.std(incs), 4) if len(incs) > 1 else None}
    print(f"{tag:<14} n={len(RTs)}  R12={np.mean(RTs):.4f}+/-{np.std(RTs):.4f}  "
          f"seeds={[round(x,3) for x in RTs]}  incorp="
          f"{(round(np.mean(incs),3) if incs else None)}")

if clean_vals:
    out["clean_ref"] = {"mean": round(np.mean(clean_vals), 4), "sd": round(np.std(clean_vals), 4),
                        "n": len(clean_vals)}
    print(f"\nclean-rebuild reference: {np.mean(clean_vals):.4f} +/- {np.std(clean_vals):.4f} "
          f"(n={len(clean_vals)})")

# trained vs anchored
if "trained" in rt and "anchored" in rt:
    d, lo, hi = boot_ci(rt["trained"], rt["anchored"])
    p = welch_p(rt["trained"], rt["anchored"])
    out["trained_vs_anchored"] = {"mean_diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                  "welch_p": round(p, 4),
                                  "verdict": ("beats" if lo > 0 else "matches (CI crosses 0)")}
    print(f"\nTRAINED - ANCHORED: {d:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  Welch p={p:.3f}  "
          f"-> {out['trained_vs_anchored']['verdict']}")

# budget-trap: parse all goodhart files
print("\n" + "=" * 70, "\nBUDGET-TRAP (R_train / R_held / gap; loose=350 vs tight=120)\n", "=" * 70)
for fp in sorted(glob.glob(os.path.join(RES, "goodhart_*.json"))):
    s = json.load(open(fp))["summary"]
    cap = s.get("word_cap") or 350
    lab = s["label"]
    is_tr = "trained" in lab
    rec = {"file": os.path.basename(fp), "cap": cap, "f": s.get("reward_frac"),
           "maintainer": "trained" if is_tr else "vanilla",
           "R_train": s.get("R_train"), "R_held": s.get("R_held"), "R_fresh": s.get("R_fresh"),
           "gap": s.get("train_minus_held"),
           "ov_train": s.get("mech_overlap_train"), "ov_held": s.get("mech_overlap_held")}
    out["trap"].append(rec)
    print(f"cap={cap:<4} f={rec['f']} {rec['maintainer']:<8} "
          f"Rtr={rec['R_train']} Rhe={rec['R_held']} gap={rec['gap']} "
          f"ov_tr/he={rec['ov_train']}/{rec['ov_held']}")

# currency
print("\n" + "=" * 70, "\nCURRENCY (displacement / entrenchment; mean +/- sd over seeds)\n", "=" * 70)
for label in ["vanilla", "b1"]:
    fs = sorted(glob.glob(os.path.join(RES, f"results_currency_{label}_seed*.json")))
    if not fs:
        continue
    disp, ent = [], []
    for f in fs:
        d = json.load(open(f))
        s = d.get("summary", d)
        disp.append(s.get("displacement_rate"))
        ent.append(s.get("entrenchment_rate"))
    disp = [x for x in disp if x is not None]; ent = [x for x in ent if x is not None]
    out["currency"][label] = {"n": len(fs),
                              "displacement": [round(np.mean(disp), 4), round(np.std(disp), 4)] if disp else None,
                              "entrenchment": [round(np.mean(ent), 4), round(np.std(ent), 4)] if ent else None}
    print(f"{label:<8} n={len(fs)} displacement={out['currency'][label]['displacement']} "
          f"entrenchment={out['currency'][label]['entrenchment']}")

# scale band
print("\n" + "=" * 70, "\nSCALE (1.5B maintainer R12 band)\n", "=" * 70)
fs = sorted(glob.glob(os.path.join(RES, "results_scale_maint1p5b_seed*.json")))
if fs:
    RTs = [_corr_RT(json.load(open(f))) for f in fs]
    out["scale"]["maint1p5b"] = {"n": len(RTs), "R12_mean": round(np.mean(RTs), 4),
                                 "R12_sd": round(np.std(RTs), 4), "per_seed": [round(x, 4) for x in RTs]}
    print(f"1.5B  n={len(RTs)}  R12={np.mean(RTs):.4f}+/-{np.std(RTs):.4f}  {[round(x,3) for x in RTs]}")
if "vanilla" in rt:
    print(f"14B vanilla R12={np.mean(rt['vanilla']):.4f}; 14B trained R12="
          f"{np.mean(rt.get('trained',[float('nan')])):.4f}")

json.dump(out, open(os.path.join(RES, "summary_v2.json"), "w"), indent=1)
print("\n-> results/summary_v2.json")
