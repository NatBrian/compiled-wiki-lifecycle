"""Trained-maintainer stage analysis — decay law, survival, hazard, and the
mitigation tournament.

Discovers arm groups from results/results_<armtag>_seed<k>.json, pools seeds per arm, and
emits results/analysis_<tag>.json containing, per arm:
  decay:    pooled corrected R(t), exponential-vs-power fit (AIC), R0, R(T).
  survival: Kaplan-Meier curve + median half-life, flicker, event/censor counts.
  hazard:   discrete-time logistic P(death|alive) ~ page_rewrites + page_words + batch.
  clean:    fresh-rebuild reference band (mean corrected at first/last checkpoint).
  cost:     mean build + maintenance generation calls (deploy cost = maint calls).
and a tournament table comparing every arm on R(T), half-life, floor, gap-recovered
relative to the `vanilla` arm and the clean band, and deploy cost.

Decay-fit / survival / hazard helpers are ported verbatim from this stage's earlier
diagnosis work (same methodology -> directly comparable numbers).
"""
import argparse, glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")  # bundled repo: no nested code/, scripts live here


# ---- ported from this stage's earlier diagnosis work (identical
#      methodology) ----------------------------------------------------------------------
def fact_timelines(run):
    tl = run["timeline"]
    claims = list(tl[0]["verdicts"].keys())
    return {c: [e["verdicts"][c] for e in tl] for c in claims}


def death_time(vs):
    last1 = max((i for i, v in enumerate(vs) if v == 1), default=-1)
    if last1 == len(vs) - 1:
        return None
    if last1 == -1:
        return 0
    return last1 + 1


def flicker_stats(series):
    n_flicker = sum(1 for vs in series.values()
                    if any(vs[i] == 0 and 1 in vs[i + 1:] for i in range(len(vs))))
    return {"n_facts": len(series), "n_flicker": n_flicker,
            "flicker_rate": round(n_flicker / max(1, len(series)), 4)}


def kaplan_meier(events):
    S, curve = 1.0, []
    for t in sorted({t for t, _ in events}):
        at_risk = sum(1 for ti, _ in events if ti >= t)
        d = sum(1 for ti, ob in events if ti == t and ob)
        if at_risk > 0:
            S *= (1 - d / at_risk)
        curve.append((t, round(S, 4)))
    median = next((t for t, s in curve if s <= 0.5), None)
    return curve, median


def fit_decay(ts, Rs):
    from scipy.optimize import curve_fit
    ts, Rs = np.array(ts, float), np.array(Rs, float)
    out = {}

    def aic(rss, k, n):
        return n * math.log(max(rss, 1e-12) / n) + 2 * k

    try:
        pe, _ = curve_fit(lambda t, r, l, c: r * np.exp(-l * t) + c, ts, Rs,
                          p0=[Rs[0], 0.05, 0.2], maxfev=20000,
                          bounds=([0, 0, 0], [1.5, 2, 1]))
        rss = float(np.sum((Rs - (pe[0] * np.exp(-pe[1] * ts) + pe[2])) ** 2))
        out["exponential"] = {"R0": round(pe[0], 4), "lam": round(pe[1], 4),
                              "floor": round(pe[2], 4), "rss": round(rss, 5),
                              "aic": round(aic(rss, 3, len(ts)), 2),
                              "halflife_from_fit": round(math.log(2) / pe[1], 2)
                              if pe[1] > 1e-6 else None}
    except Exception as e:
        out["exponential"] = {"error": str(e)}
    try:
        pp, _ = curve_fit(lambda t, r, a, c: r * (1 + t) ** (-a) + c, ts, Rs,
                          p0=[Rs[0], 0.5, 0.2], maxfev=20000,
                          bounds=([0, 0, 0], [1.5, 5, 1]))
        rss = float(np.sum((Rs - (pp[0] * (1 + ts) ** (-pp[1]) + pp[2])) ** 2))
        out["power"] = {"R0": round(pp[0], 4), "alpha": round(pp[1], 4),
                        "floor": round(pp[2], 4), "rss": round(rss, 5),
                        "aic": round(aic(rss, 3, len(ts)), 2)}
    except Exception as e:
        out["power"] = {"error": str(e)}
    if "aic" in out.get("exponential", {}) and "aic" in out.get("power", {}):
        out["preferred"] = ("exponential" if out["exponential"]["aic"] <= out["power"]["aic"]
                            else "power")
    return out


def hazard_model(runs):
    import pandas as pd
    import statsmodels.api as sm
    rows = []
    for ri, run in enumerate(runs):
        pmap = run.get("page_of_probe")
        if not pmap:
            return None
        tl = run["timeline"]
        for claim, vs in fact_timelines(run).items():
            dt = death_time(vs)
            if dt == 0:
                continue
            horizon = dt if dt is not None else len(vs) - 1
            pi = pmap[claim]
            for t in range(1, horizon + 1):
                e = tl[t]
                if pi >= len(e["rewrites"]):
                    continue
                rows.append({"event": 1 if (dt is not None and t == dt) else 0,
                             "page_rewrites": e["rewrites"][pi],
                             "page_words": e["page_words"][pi], "batch": t, "seed": ri})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    X = sm.add_constant(df[["page_rewrites", "page_words", "batch"]].astype(float))
    try:
        fit = sm.Logit(df["event"], X).fit(disp=0)
        return {"n_cells": len(df), "n_events": int(df["event"].sum()),
                "coef": {k: round(v, 4) for k, v in fit.params.items()},
                "pvalues": {k: round(v, 4) for k, v in fit.pvalues.items()}}
    except Exception as e:
        return {"error": str(e), "n_cells": len(df)}


# ---- additions specific to this stage --------------------------------------------------
def clean_band(runs):
    """Mean corrected retention of the fresh-rebuild reference, std flavor, per checkpoint t."""
    by_t = {}
    for r in runs:
        for cp in r.get("checkpoints", []):
            std = cp["clean"].get("std")
            if std and "corrected_mean" in std:
                by_t.setdefault(cp["t"], []).append(std["corrected_mean"])
    return {t: round(float(np.mean(v)), 4) for t, v in sorted(by_t.items())}


def _ret(entry):
    """Corrected retention if present (SciFact schema), else raw string-match (HoH schema)."""
    m = entry.get("metrics")
    return m["corrected"] if m else entry.get("raw")


def analyze_arm(runs):
    T = len(runs[0]["timeline"])
    per_seed = [[_ret(r["timeline"][t]) for t in range(T)] for r in runs]
    mean_R = [round(float(np.mean([s[t] for s in per_seed])), 4) for t in range(T)]
    sd_R = [round(float(np.std([s[t] for s in per_seed])), 4) for t in range(T)]

    events, flick, never = [], [], 0
    for r in runs:
        series = fact_timelines(r)
        flick.append(flicker_stats(series))
        for vs in series.values():
            dt = death_time(vs)
            if dt == 0:
                never += 1
            elif dt is None:
                events.append((T - 1, False))
            else:
                events.append((dt, True))
    curve, median = kaplan_meier(events)

    band = clean_band(runs)
    gc = [r.get("gen_calls", {}) for r in runs]
    cost_maint = round(float(np.mean([g.get("maint", 0) for g in gc])), 1)
    cost_build = round(float(np.mean([g.get("build", 0) for g in gc])), 1)
    # incorporation rate (does the maintainer actually absorb new docs, or cheat by ignoring them?)
    inc = []
    for r in runs:
        irs = [e["incorp_rate"] for e in r["timeline"] if e.get("incorp_rate") is not None]
        if irs:
            inc.append(float(np.mean(irs)))
    incorp = round(float(np.mean(inc)), 4) if inc else None
    incorp_sd = round(float(np.std(inc)), 4) if len(inc) > 1 else None
    incorp_per_seed = [round(x, 4) for x in inc] if inc else None

    return {
        "n_seeds": len(runs), "T": T - 1,
        "per_seed_R": per_seed, "mean_R": mean_R, "sd_R": sd_R,
        "R0": mean_R[0], "RT": mean_R[-1],
        "fit": fit_decay(list(range(T)), mean_R),
        "survival": {"km_curve": curve, "median_halflife_batches": median,
                     "n_events": sum(1 for _, ob in events if ob),
                     "n_censored": sum(1 for _, ob in events if not ob),
                     "n_never_t0": never, "flicker_per_seed": flick},
        "hazard": hazard_model(runs) or "skipped",
        "clean_band": band,
        "incorp_rate": incorp, "incorp_sd": incorp_sd, "incorp_per_seed": incorp_per_seed,
        "cost": {"maint_calls": cost_maint, "build_calls": cost_build,
                 "deploy_calls": cost_maint},
    }


def discover_arms(pattern):
    """results_<armtag>_seed<k>.json -> {armtag: [files]}. pattern may be comma-separated globs."""
    arms = {}
    for pat in pattern.split(","):
        for f in sorted(glob.glob(os.path.join(RES, pat.strip()))):
            m = re.match(r"results_(.+)_seed(\d+)\.json$", os.path.basename(f))
            if m:
                arms.setdefault(m.group(1), []).append(f)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results_a1_*_seed*.json")
    ap.add_argument("--tag", default="a1")
    ap.add_argument("--baseline", default=None,
                    help="armtag treated as the disease baseline for gap-recovered "
                         "(default: any tag containing 'vanilla' or 'naive')")
    args = ap.parse_args()

    arms = discover_arms(args.pattern)
    if not arms:
        sys.exit(f"no runs matched {args.pattern}")
    report = {"arms": {}, "pattern": args.pattern}
    for tag, files in arms.items():
        runs = [json.load(open(f)) for f in files]
        report["arms"][tag] = analyze_arm(runs)
        report["arms"][tag]["files"] = [os.path.basename(f) for f in files]

    # baseline + clean reference for gap-recovered
    base_tag = args.baseline or next((t for t in arms if "vanilla" in t or "naive" in t), None)
    base_RT = report["arms"][base_tag]["RT"] if base_tag else None
    # clean reference = max-t clean band averaged over all arms (same fresh rebuild)
    all_bands = [a["clean_band"] for a in report["arms"].values() if a["clean_band"]]
    clean_ref = None
    if all_bands:
        maxt = max(max(b) for b in all_bands)
        clean_ref = round(float(np.mean([b[maxt] for b in all_bands if maxt in b])), 4)

    table = []
    for tag, a in report["arms"].items():
        gap = None
        if base_RT is not None and clean_ref is not None and clean_ref > base_RT:
            gap = round((a["RT"] - base_RT) / (clean_ref - base_RT), 3)
        ex = a["fit"].get("exponential", {})
        table.append({"arm": tag, "R0": a["R0"], "RT": a["RT"], "sd_RT": a["sd_R"][-1],
                      "halflife_km": a["survival"]["median_halflife_batches"],
                      "halflife_fit": ex.get("halflife_from_fit"),
                      "floor": ex.get("floor"), "lam": ex.get("lam"),
                      "incorp_rate": a["incorp_rate"], "incorp_sd": a.get("incorp_sd"),
                      "deploy_calls": a["cost"]["deploy_calls"],
                      "gap_recovered": gap, "n_seeds": a["n_seeds"]})
    table.sort(key=lambda r: (r["RT"] is None, -(r["RT"] or 0)))
    report["tournament"] = {"baseline_arm": base_tag, "baseline_RT": base_RT,
                            "clean_ref_RT": clean_ref, "table": table}

    out = os.path.join(RES, f"analysis_{args.tag}.json")
    json.dump(report, open(out, "w"), indent=1)
    print("=== TOURNAMENT (clean_ref RT =", clean_ref, ", baseline", base_tag,
          "RT =", base_RT, ") ===")
    hdr = f"{'arm':<26}{'R0':>7}{'RT':>7}{'sdRT':>7}{'t½km':>7}{'t½fit':>7}{'floor':>7}{'gap%':>7}{'cost':>7}"
    print(hdr)
    for r in table:
        print(f"{r['arm']:<26}{r['R0']:>7}{r['RT']:>7}{r['sd_RT']:>7}"
              f"{str(r['halflife_km']):>7}{str(r['halflife_fit']):>7}{str(r['floor']):>7}"
              f"{str(r['gap_recovered']):>7}{str(r['deploy_calls']):>7}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
