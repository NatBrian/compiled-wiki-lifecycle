"""Anytime-valid composition of per-batch loss budgets into a stream-level retention LCB.

Given per-batch certified loss bounds {b_t} (each R(t) >= R(t-1) - b_t at level 1-alpha)
and a t=0 certificate L_0 (one-sided LCB on R(0) at 1-alpha), we want a guarantee that
holds SIMULTANEOUSLY over an unbounded maintenance stream:

    R(t) >= L_0 - sum_{s=1..t} b_t^*    for all t, with overall error <= delta.

Two compositions are provided:

1. union_bound: split delta across the horizon (alpha_t = delta / (T+1)); each component
   recomputed at its own alpha. Simple, valid for a FIXED horizon T.

2. confidence_sequence: a time-uniform (anytime-valid) bound that never needs T fixed.
   Each b_t is recomputed with a per-step level that sums to delta via a converging
   series alpha_t = delta * 6/(pi^2 t^2) (Basel weights), so the union over ALL t<=infty
   has error <= delta. This is the e-process / confidence-sequence flavor: the stream LCB
   L(t) = L_0 - cumsum(b_t) is valid at every t simultaneously.

The realized stream retention (measured on the held-out VALID pool) must stay >= L(t)
for the guarantee to be empirically valid (E2). Compared against P2's STATIC certificate,
which is L_0 forever and breaches once true retention decays below it.
"""
import math
from stats import cp_upper


def basel_weights(T):
    """alpha_t proportional to 1/t^2, summing to 1 over t=1..inf (pi^2/6 normalization)."""
    return [6.0 / (math.pi ** 2 * t * t) for t in range(1, T + 1)]


def recompute_b(d, n, alpha, tpr, fpr):
    """Per-batch loss bound at a given alpha: judge-corrected CP-upper on destroyed fraction."""
    if n == 0:
        return 0.0
    b = cp_upper(d, n, alpha)
    margin = tpr - fpr
    return min(1.0, b / margin) if margin > 0 else 1.0


def stream_lcb(per_batch, L0, delta, tpr, fpr, mode="confidence_sequence"):
    """per_batch = list of dicts with keys d (destroyed), n (confirmed_prev) per t=1..T.
    Returns list of stream LCBs L(t) for t=0..T (L(0)=L0), recomputing each b_t at its
    composition-adjusted level so the union over the horizon has total error <= delta."""
    T = len(per_batch)
    if mode == "union_bound":
        alphas = [delta / (T + 1)] * T
    elif mode == "confidence_sequence":
        w = basel_weights(T)
        alphas = [delta * wi for wi in w]  # sum_t alpha_t <= delta
    else:
        raise ValueError(mode)
    out = [L0]
    acc = 0.0
    for t, pb in enumerate(per_batch):
        acc += recompute_b(pb["d"], pb["n"], alphas[t], tpr, fpr)
        out.append(max(0.0, L0 - acc))
    return out, alphas


if __name__ == "__main__":
    # self-check on synthetic d/n
    pb = [{"d": 0, "n": 30}, {"d": 1, "n": 30}, {"d": 0, "n": 29}, {"d": 2, "n": 28}]
    L, a = stream_lcb(pb, L0=0.6, delta=0.05, tpr=0.9, fpr=0.1)
    print("stream LCB:", [round(x, 3) for x in L])
    print("alphas:", [round(x, 4) for x in a])
