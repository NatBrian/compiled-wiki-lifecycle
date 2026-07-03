"""Certificate statistics: judge-noise-corrected lower confidence bound on retention.

Model: judge verdict V in {0,1} per probe; true presence S in {0,1}.
  E[V] = R*TPR + (1-R)*FPR  =>  R = (E[V]-FPR)/(TPR-FPR)
R is increasing in p=E[V], decreasing in TPR and FPR (when FPR < p < TPR).
Conservative one-sided LCB at level 1-alpha by union bound (each component CI
at alpha/3): LCB(R) = (p_L - fpr_U) / (tpr_U - fpr_U), clipped to [0,1].
Clopper-Pearson (exact) one-sided intervals throughout.
"""
import math

from scipy.stats import beta


def cp_lower(k, n, alpha):
    """One-sided exact lower bound for binomial proportion."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return beta.ppf(alpha, k, n - k + 1)


def cp_upper(k, n, alpha):
    """One-sided exact upper bound for binomial proportion."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return beta.ppf(1 - alpha, k + 1, n - k)


def coverage_simulation(R_true, tpr, fpr, n_audit, n_tpr, n_fpr, alpha=0.05,
                        n_sims=4000, seed=0):
    """Proper coverage check: simulate verdicts from a KNOWN true retention R_true
    under the judge-noise model, recompute the certificate from simulated audit AND
    simulated calibration draws, and measure P(LCB <= R_true). A valid 1-alpha bound
    should cover >= 1-alpha. (Replaces the earlier circular self-referential check.)"""
    import random as _r
    rng = _r.Random(seed)
    p_obs = R_true * tpr + (1 - R_true) * fpr
    hits = 0
    for _ in range(n_sims):
        kv = sum(rng.random() < p_obs for _ in range(n_audit))
        kt = sum(rng.random() < tpr for _ in range(n_tpr))
        kf = sum(rng.random() < fpr for _ in range(n_fpr))
        lcb, _ = retention_lcb(kv, n_audit, kt, n_tpr, kf, n_fpr, alpha=alpha)
        hits += lcb <= R_true + 1e-9
    return hits / n_sims


def corrected_point(p, tpr, fpr):
    """Point estimate of true retention from observed verdict rate."""
    den = tpr - fpr
    if den <= 0:
        return float("nan")
    return float(min(1.0, max(0.0, (p - fpr) / den)))


def retention_lcb(k_v, n_v, k_tpr, n_tpr, k_fpr, n_fpr, alpha=0.05):
    """Conservative LCB on true retention R.

    k_v/n_v: judge-YES count on audit probes (store contexts).
    k_tpr/n_tpr: judge-YES count on fact-PRESENT calibration pairs.
    k_fpr/n_fpr: judge-YES count on fact-ABSENT calibration pairs.
    Union bound: alpha/3 per component, one-sided in the conservative direction.
    """
    a = alpha / 3
    p_l = cp_lower(k_v, n_v, a)
    tpr_u = cp_upper(k_tpr, n_tpr, a)
    fpr_u = cp_upper(k_fpr, n_fpr, a)
    den = tpr_u - fpr_u
    if den <= 0.05:  # judge uninformative at this confidence
        return 0.0, {"p_l": round(float(p_l),4), "tpr_u": round(float(tpr_u),4), "fpr_u": round(float(fpr_u),4), "degenerate": True}
    lcb = (p_l - fpr_u) / den
    # p_l > tpr_u means the observed audit rate exceeds the upper TPR bound (model
    # regime broken); clipping up to 1.0 would silently assert certainty -> flag it.
    overshoot = bool(lcb > 1.0)
    return float(min(1.0, max(0.0, lcb))), {"p_l": round(float(p_l), 4), "tpr_u": round(float(tpr_u), 4),
                                     "fpr_u": round(float(fpr_u), 4), "degenerate": False,
                                     "overshoot": overshoot}


def ser_upper(k_v, n_v, k_tpr, n_tpr, k_fpr, n_fpr, alpha=0.05):
    """Judge-noise-corrected UPPER bound on true staleness SER.

    Stale-answer judge: V=1 if store yields the superseded answer. With
    observed rate p = SER*TPR + (1-SER)*FPR, SER = (p-FPR)/(TPR-FPR) is
    increasing in p, decreasing in TPR and FPR. Upper bound maximizes p
    (p_U), minimizes TPR (tpr_L) and FPR (fpr_L). Union bound alpha/3.
      k_tpr/n_tpr: stale-context calibration (stale present -> should yield stale).
      k_fpr/n_fpr: current-context calibration (stale absent -> should NOT yield stale).
    """
    a = alpha / 3
    p_u = cp_upper(k_v, n_v, a)
    tpr_l = cp_lower(k_tpr, n_tpr, a)
    fpr_l = cp_lower(k_fpr, n_fpr, a)
    den = tpr_l - fpr_l
    if den <= 0.05:
        return 1.0, {"p_u": round(float(p_u),4), "tpr_l": round(float(tpr_l),4), "fpr_l": round(float(fpr_l),4), "degenerate": True}
    ub = (p_u - fpr_l) / den
    return float(min(1.0, max(0.0, ub))), {"p_u": round(float(p_u), 4), "tpr_l": round(float(tpr_l), 4),
                                    "fpr_l": round(float(fpr_l), 4), "degenerate": False}


def stratified_lcb(strata, alpha=0.05):
    """LCB on weighted retention across page strata (for incremental maintenance).

    strata: list of dicts {w, k_v, n_v} sharing global TPR/FPR counts passed
    separately is handled by caller; here a simple weighted CP composition:
    bound each stratum verdict rate at alpha/(2*S) and combine linearly.
    (Used in E2; conservative.)
    """
    tot_w = sum(s["w"] for s in strata)
    a = alpha / max(1, len(strata))
    lo = 0.0
    for s in strata:
        lo += (s["w"] / tot_w) * cp_lower(s["k_v"], s["n_v"], a)
    return lo


def wilson_lower(k, n, alpha):
    """Wilson one-sided lower bound (reference / comparisons only)."""
    if n == 0:
        return 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha)
    p = k / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - rad) / den)
