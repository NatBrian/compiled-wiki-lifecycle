"""Cluster-robust statistics for the paper spine (Block 1 MUST-RUN stats target).

Each fact contributes one binary `backflowed` outcome PER ARM -> observations are
correlated within fact_id. Wilson CIs (used in the merge) ignore that. Here we:

  1. GEE (Binomial, exchangeable, groups=fact_id) -> cluster-robust RSR + 95% CI
     per arm; report ICC and design effect DEFF = 1 + (m_bar - 1)*ICC and the
     effective sample size n_eff = n / DEFF.
  2. TOST equivalence of the best gating rung (A3) vs the never-ingested oracle
     (A4) on the cluster-robust risk-difference at margin Delta=0.05, plus a
     cluster bootstrap (resample fact_ids, B=10000) two-sided 90% CI on the RD
     (90% CI within +/-Delta <=> TOST rejects non-equivalence at alpha=0.05).
  3. Same machinery for E0b-AUTO (A1_auto vs A3_auto: discrimination; A3_auto vs
     A4: equivalence-to-oracle).

Reads results/b1_ladder.json and results/e0b_auto.json (whatever n is on disk).
Writes results/stats_cluster.json. No GPU.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = 0.05      # equivalence margin (risk difference)
B_BOOT = 10000    # cluster bootstrap resamples
SEED = 20260620


def long_df(path, arms):
    """Build long format: rows = (fact_id, arm, y) where y = backflowed (0/1)."""
    d = json.load(open(path))["per_fact"]
    rows = []
    for r in d:
        for a in arms:
            if a in r and isinstance(r[a], dict) and "backflowed" in r[a]:
                rows.append({"fid": r["id"], "arm": a, "y": int(bool(r[a]["backflowed"]))})
    return pd.DataFrame(rows)


def icc_deff(df):
    """ICC via one-way random effects on the binary outcome across arms within
    fact; DEFF = 1 + (m_bar - 1) * ICC."""
    g = df.groupby("fid")["y"]
    m = g.count()
    m_bar = m.mean()
    grand = df["y"].mean()
    # ANOVA-style ICC for binary (Fleiss-style): between vs within fact variance
    msb = g.mean().sub(grand).pow(2).mul(m).sum() / max(len(m) - 1, 1)
    msw = df.assign(gm=df.groupby("fid")["y"].transform("mean")) \
            .eval("(y - gm)**2").sum() / max(len(df) - len(m), 1)
    denom = msb + (m_bar - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    icc = float(max(0.0, min(1.0, icc)))
    deff = 1 + (m_bar - 1) * icc
    return icc, float(deff), float(m_bar)


def gee_rsr_per_arm(df):
    """Intercept-only GEE per arm -> cluster-robust RSR (mean) + 95% CI.
    GEE on a single arm is degenerate (one obs/cluster) so we report the
    cluster-robust proportion via the multi-arm model's marginal means instead."""
    out = {}
    arms = sorted(df["arm"].unique())
    # multi-arm GEE: y ~ C(arm), exchangeable within fid -> robust SEs
    try:
        m = smf.gee("y ~ C(arm)", "fid", data=df,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        # predict marginal prob per arm
        for a in arms:
            sub = pd.DataFrame({"arm": [a]})
            p = float(m.predict(sub).iloc[0])
            out[a] = {"RSR_gee": round(p, 4)}
    except Exception as e:  # noqa: BLE001
        for a in arms:
            out[a] = {"RSR_gee": round(float(df[df.arm == a]["y"].mean()), 4),
                      "gee_note": f"gee_failed:{str(e)[:60]}"}
    # raw proportion + naive Wilson for reference
    for a in arms:
        sub = df[df.arm == a]["y"]
        out[a]["RSR_raw"] = round(float(sub.mean()), 4)
        out[a]["n"] = int(sub.shape[0])
    return out


def cluster_bootstrap_rd(df, arm_a, arm_b, conf=0.90):
    """Two-sided (conf) CI on RD = P(y|arm_a) - P(y|arm_b) by resampling fact_ids."""
    rng = np.random.default_rng(SEED)
    piv = df.pivot_table(index="fid", columns="arm", values="y", aggfunc="first")
    piv = piv.dropna(subset=[arm_a, arm_b])
    A = piv[arm_a].to_numpy(); Bv = piv[arm_b].to_numpy()
    n = len(A)
    point = float(A.mean() - Bv.mean())
    diffs = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = A[idx].mean() - Bv[idx].mean()
    lo = float(np.quantile(diffs, (1 - conf) / 2))
    hi = float(np.quantile(diffs, 1 - (1 - conf) / 2))
    return point, lo, hi, n


def tost(df, arm_test, arm_ref, label):
    point, lo, hi, n = cluster_bootstrap_rd(df, arm_test, arm_ref, conf=0.90)
    equivalent = (lo > -DELTA) and (hi < DELTA)  # 90% CI inside +/-Delta
    return {"comparison": f"{arm_test} vs {arm_ref}", "label": label,
            "rd_point": round(point, 4),
            "rd_90ci_cluster_boot": [round(lo, 4), round(hi, 4)],
            "equivalence_margin": DELTA, "n_pairs": n,
            "TOST_equivalent": bool(equivalent)}


def analyze(path, arms, disc_pair, equiv_pair, name):
    df = long_df(path, arms)
    icc, deff, m_bar = icc_deff(df)
    per_arm = gee_rsr_per_arm(df)
    n_total = len(df)
    res = {"source": os.path.basename(path), "n_facts": int(df["fid"].nunique()),
           "arms": arms, "per_arm": per_arm,
           "ICC": round(icc, 4), "DEFF": round(deff, 3), "m_bar": round(m_bar, 2),
           "n_obs": n_total, "n_eff": round(n_total / deff, 1) if deff else None}
    # discrimination: test vs ref should be NON-equivalent (big RD)
    dp, dlo, dhi, dn = cluster_bootstrap_rd(df, disc_pair[0], disc_pair[1], conf=0.95)
    res["discrimination"] = {"comparison": f"{disc_pair[0]} vs {disc_pair[1]}",
                             "rd_point": round(dp, 4),
                             "rd_95ci_cluster_boot": [round(dlo, 4), round(dhi, 4)],
                             "ci_excludes_0": bool(dlo > 0 or dhi < 0)}
    # equivalence to oracle
    res["equivalence_to_oracle"] = tost(df, equiv_pair[0], equiv_pair[1],
                                        "gated rung equivalent to never-ingested oracle")
    return res


def main():
    out = {"delta": DELTA, "B_boot": B_BOOT}
    b1 = f"{ROOT}/results/b1_ladder.json"
    if os.path.exists(b1):
        out["B1"] = analyze(b1, ["A0", "A1", "A2", "A3", "A4"],
                            disc_pair=("A2", "A3"), equiv_pair=("A3", "A4"),
                            name="B1")
    au = f"{ROOT}/results/e0b_auto.json"
    if os.path.exists(au):
        out["AUTO"] = analyze(au, ["A1_auto", "A3_auto", "A4"],
                             disc_pair=("A1_auto", "A3_auto"),
                             equiv_pair=("A3_auto", "A4"), name="AUTO")
    json.dump(out, open(f"{ROOT}/results/stats_cluster.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
