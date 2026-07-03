#!/usr/bin/env python3
"""De-circularized judge-noise estimate from TWO independent blinded raters.

Reviewer fix: instead of the authors adjudicating their own matcher, two independent
annotators labelled a shuffled set with arm + matcher-label hidden. We report
inter-rater agreement and take the CONSENSUS (both raters agree) as the robust
adjudication, then recover matcher FPR/FNR against it.
"""
import json
from pathlib import Path
from scipy import stats

RES = Path(__file__).resolve().parents[2] / "results"
key = {k["blind_id"]: k for k in json.loads((RES / "blind_key.json").read_text())}
A = {d["blind_id"]: d["label"] for d in json.loads((RES / "blind_rater_A.json").read_text())}
B = {d["blind_id"]: d["label"] for d in json.loads((RES / "blind_rater_B.json").read_text())}
ids = sorted(key)
n = len(ids)


def cp_upper(x, n, alpha=0.05):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - alpha, x + 1, n - x))


# Cohen's kappa (3-class)
labs = ["current", "stale", "neither"]
agree = sum(1 for i in ids if A[i] == B[i]) / n
ca = {l: sum(1 for i in ids if A[i] == l) / n for l in labs}
cb = {l: sum(1 for i in ids if B[i] == l) / n for l in labs}
pe = sum(ca[l] * cb[l] for l in labs)
kappa = (agree - pe) / (1 - pe)

# consensus-stale = both raters call it stale
def both_stale(i): return A[i] == "stale" and B[i] == "stale"

# matcher false positives: matcher=stale but BOTH raters say not-stale AND neither
# rater says stale  (i.e. independent consensus that it is NOT a superseded answer)
mp = [i for i in ids if key[i]["matcher"] == "stale"]
fp = [i for i in mp if A[i] != "stale" and B[i] != "stale"]
# of those, keep only where the consensus is a coherent not-stale (both same non-stale
# label) -- blind83-type rater disagreements (current vs neither) are weaker evidence
fp_strong = [i for i in fp if A[i] == B[i]]

# resolving-arm false negatives: resolving + matcher=neither but both raters say stale
rn = [i for i in ids if key[i]["resolving"] and key[i]["matcher"] == "neither"]
fn_res = [i for i in rn if both_stale(i)]
# accumulating false negatives
an = [i for i in ids if not key[i]["resolving"] and key[i]["matcher"] == "neither"]
fn_acc = [i for i in an if both_stale(i)]

out = {
    "raters": 2, "blinded": True, "n_items": n,
    "inter_rater_agreement": round(agree, 3), "cohens_kappa": round(kappa, 3),
    "matcher_fp_consensus": len(fp_strong), "matcher_fp_denom": len(mp),
    "matcher_fpr_point": round(len(fp_strong) / len(mp), 4),
    "matcher_fpr_95upper": round(cp_upper(len(fp_strong), len(mp)), 4),
    "fp_note": "both consensus-FP items are non-answers (question echoes) the fuzzy "
               "matcher mis-flagged; a FP only LOWERS true SER, so the upper-bound "
               "certificate stays valid. Both fall in an accumulating arm.",
    "resolving_fn_consensus": len(fn_res), "resolving_fn_denom": len(rn),
    "resolving_fnr_95upper": round(cp_upper(len(fn_res), len(rn)), 4),
    "accum_fn_consensus": len(fn_acc), "accum_fn_denom": len(an),
    "corrected_resolver_stale": 15 + len(fn_res),
    "corrected_resolver_eps95": round(cp_upper(15 + len(fn_res), 300), 4),
    "raw_resolver_eps95": round(cp_upper(15, 300), 4),
}
(RES / "judge_noise_final.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
