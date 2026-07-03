"""Trained-maintainer stage: mechanism of fact death: strengthened discrete-time hazard
model.

Answers WHY facts die and WHAT training changes mechanistically, from existing run logs
(zero GPU). Builds a person-period (claim x batch) panel up to each fact's death/censoring
and fits a discrete-time logistic hazard with:

  covariates (per claim x batch):
    page_rewrites : cumulative rewrites of the fact's HOST page at this batch (the damage clock)
    page_words    : host-page length (budget crowding)
    batch         : elapsed maintenance time (to dissociate rewriting from mere time)
    claim_len     : #words in the probe claim (a proxy for detail)
    claim_spec    : #numbers + #capitalized-entity tokens in the claim (specificity)
    page_crowd    : #probe facts sharing the host page at t0 (competition for budget)

  reported:
    - per-arm coefficients with CLUSTER-ROBUST SE (clustered by host page)
    - the rewrite-count x arm INTERACTION (does training attenuate the per-rewrite hazard?)
      fit on the pooled vanilla+<arm> panel
    - half-life RE-DENOMINATED IN REWRITES (deployment-invariant clock): from the fitted
      per-rewrite hazard h, expected rewrites to 50% survival = ln(0.5)/ln(1-h_marginal).

Usage:
  python hazard_analysis.py --arms a1_vanilla,b1_trained --tag v1
  python hazard_analysis.py --pattern "results_v2_*_seed*.json" --tag v2
"""
import argparse, glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")  # bundled repo: no nested code/, scripts live here

_NUM = re.compile(r"\d")
_CAP = re.compile(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b")


def claim_features(claim):
    toks = claim.split()
    n_num = sum(1 for w in toks if _NUM.search(w))
    n_cap = len(_CAP.findall(claim))
    return {"claim_len": len(toks), "claim_spec": n_num + n_cap}


def fact_timelines(run):
    tl = run["timeline"]
    claims = list(tl[0]["verdicts"].keys())
    return {c: [e["verdicts"][c] for e in tl] for c in claims}


def death_time(vs):
    last1 = max((i for i, v in enumerate(vs) if v == 1), default=-1)
    if last1 == len(vs) - 1:
        return None          # censored (alive at end)
    if last1 == -1:
        return 0             # never present (dropped from t0; excluded)
    return last1 + 1


def build_panel(runs, arm):
    """Person-period rows for one arm's runs (pooled over seeds)."""
    rows = []
    for ri, run in enumerate(runs):
        pmap = run.get("page_of_probe")
        if not pmap:
            continue
        tl = run["timeline"]
        # page crowding at t0: probes per host page
        crowd = {}
        for c, pi in pmap.items():
            crowd[pi] = crowd.get(pi, 0) + 1
        feats = {c["claim"]: claim_features(c["claim"]) for c in run["probes"]}
        # map probe-claim text -> its claim feature (verdict keys are claim text)
        for claim, vs in fact_timelines(run).items():
            dt = death_time(vs)
            if dt == 0:
                continue
            horizon = dt if dt is not None else len(vs) - 1
            pi = pmap.get(claim)
            if pi is None:
                continue
            f = feats.get(claim, {"claim_len": len(claim.split()), "claim_spec": 0})
            for t in range(1, horizon + 1):
                e = tl[t]
                if pi >= len(e["rewrites"]):
                    continue
                rows.append({
                    "arm": arm, "seed": ri, "page": f"{ri}:{pi}", "claim": f"{ri}:{claim[:40]}",
                    "event": 1 if (dt is not None and t == dt) else 0,
                    "page_rewrites": e["rewrites"][pi],
                    "page_words": e["page_words"][pi],
                    "batch": t,
                    "claim_len": f["claim_len"], "claim_spec": f["claim_spec"],
                    "page_crowd": crowd.get(pi, 1),
                })
    return rows


def fit_logit(df, cols, cluster=None):
    import statsmodels.api as sm
    X = sm.add_constant(df[cols].astype(float))
    y = df["event"].astype(float)
    model = sm.Logit(y, X)
    if cluster is not None:
        fit = model.fit(disp=0, cov_type="cluster",
                        cov_kwds={"groups": df[cluster]})
    else:
        fit = model.fit(disp=0)
    out = {}
    ci = fit.conf_int()
    for k in X.columns:
        out[k] = {"coef": round(float(fit.params[k]), 4),
                  "se": round(float(fit.bse[k]), 4),
                  "p": round(float(fit.pvalues[k]), 4),
                  "ci": [round(float(ci.loc[k, 0]), 4), round(float(ci.loc[k, 1]), 4)]}
    return out, fit


def marginal_hazard_per_rewrite(df, fit_params):
    """Average marginal P(death) increase per additional host-page rewrite (discrete diff at
    the panel mean), and the implied rewrite-clock half-life."""
    import statsmodels.api as sm
    cols = [c for c in fit_params.index if c != "const"]
    mean = {c: df[c].astype(float).mean() for c in cols}

    def p(rw):
        z = fit_params["const"] + sum(fit_params[c] * (rw if c == "page_rewrites" else mean[c])
                                      for c in cols)
        return 1 / (1 + math.exp(-z))
    lo = max(0, int(round(mean["page_rewrites"])) - 1)
    h0, h1 = p(lo), p(lo + 1)
    h = max(1e-6, (h0 + h1) / 2)
    halflife_rewrites = math.log(0.5) / math.log(1 - h) if h < 1 else None
    return {"h_at_mean": round(h, 4),
            "dP_per_rewrite": round(h1 - h0, 4),
            "halflife_rewrites": round(halflife_rewrites, 2) if halflife_rewrites else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="comma list of armtags (results_<arm>_seed*.json)")
    ap.add_argument("--pattern", default=None, help="alternative: glob(s) for results files")
    ap.add_argument("--baseline", default=None, help="armtag used as the disease baseline for the interaction (default: a *vanilla* arm)")
    ap.add_argument("--tag", default="hazard")
    args = ap.parse_args()

    import pandas as pd

    # discover arm -> files
    arm_files = {}
    if args.arms:
        for arm in args.arms.split(","):
            arm = arm.strip()
            fs = sorted(glob.glob(os.path.join(RES, f"results_{arm}_seed*.json")))
            if fs:
                arm_files[arm] = fs
    if args.pattern:
        for pat in args.pattern.split(","):
            for f in sorted(glob.glob(os.path.join(RES, pat.strip()))):
                m = re.match(r"results_(.+)_seed\d+\.json$", os.path.basename(f))
                if m:
                    arm_files.setdefault(m.group(1), []).append(f)
    if not arm_files:
        sys.exit("no arms found")

    cols = ["page_rewrites", "page_words", "batch", "claim_len", "claim_spec", "page_crowd"]
    report = {"arms": {}, "interaction": {}}
    panels = {}
    print(f"=== HAZARD (cluster-robust by page) — covariates: {cols} ===")
    for arm, files in arm_files.items():
        runs = [json.load(open(f)) for f in files]
        rows = build_panel(runs, arm)
        if not rows:
            print(f"{arm}: no panel rows"); continue
        df = pd.DataFrame(rows)
        panels[arm] = df
        try:
            coefs, fit = fit_logit(df, cols, cluster="page")
            mh = marginal_hazard_per_rewrite(df, fit.params)
        except Exception as e:
            print(f"{arm}: fit error {e}"); continue
        report["arms"][arm] = {"n_cells": len(df), "n_events": int(df["event"].sum()),
                               "n_seeds": len(files), "coef": coefs, "rewrite_clock": mh}
        pr = coefs["page_rewrites"]
        print(f"{arm:<22} n={len(df):<6} ev={int(df['event'].sum()):<4} "
              f"page_rewrites coef={pr['coef']:+.4f} (p={pr['p']}, 95%CI {pr['ci']}) | "
              f"t½={mh['halflife_rewrites']} rewrites")

    # rewrite-count x arm interaction: pool baseline vanilla + each other arm
    base = args.baseline or next((a for a in arm_files if "vanilla" in a), None)
    if base and base in panels:
        for arm in panels:
            if arm == base:
                continue
            d = pd.concat([panels[base].assign(is_arm=0.0), panels[arm].assign(is_arm=1.0)],
                          ignore_index=True)
            d["rw_x_arm"] = d["page_rewrites"] * d["is_arm"]
            try:
                coefs, _ = fit_logit(d, cols + ["is_arm", "rw_x_arm"], cluster="page")
                inter = coefs["rw_x_arm"]
                report["interaction"][f"{base}_vs_{arm}"] = coefs
                print(f"INTERACTION {base} vs {arm}: rewrite-count x arm = "
                      f"{inter['coef']:+.4f} (p={inter['p']}, 95%CI {inter['ci']}) "
                      f"[negative => {arm} lowers the per-rewrite hazard]")
            except Exception as e:
                print(f"interaction {base} vs {arm}: {e}")

    out = os.path.join(RES, f"hazard_{args.tag}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
