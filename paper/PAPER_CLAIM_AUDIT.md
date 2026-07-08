# Paper Claim Audit

**Date**: 2026-07-03
**Method**: every numeric claim in `main.tex`/`sections/*.tex` checked directly against the underlying results, logs, and analysis scripts for each stage's experiments.

## Result: PASS, zero numeric drift found

Checked and confirmed matching the underlying results, section by section:

- **Compile**: all 7 SER/accuracy arm values, the 2×2 structure×resolution dissociation, both certificate bounds ($\hat\varepsilon$), and the external-validation table (`r0`/`rT`/rebuild for all 3 arms, relative-retention percentages recomputed and checked: 13.3/17.3=76.9%→77%, 29.3/32=91.6%→92%, 2.7/22.7=11.9%→12%).
- **Certify**: E1 gate numbers (raw/corrected/LCB for both wiki and RAG), price-of-certification curve endpoints, build-lottery percentages and the A/B example pair, E2/E2b cost and breach numbers, E4 adaptivity-trap gap and the compile-twice-union honest gain, E5 born-stale→maintained bounds.
- **Maintain**: the trained-maintainer's $R_{12}$ values with seed counts and CI, the Welch $p$-values (0.009 real, 0.54 null, both carried through correctly, the null is stated as null, not rounded into a claim), the boundary-law signal/policy numbers at both budget levels and the ~6× ratio with its noise-floor caveat. LossGate's per-arm $R(12)$ table, paired-gain sigma values, coverage numbers, the small-pool-vs-large-pool bound tightness contrast, the E4-LoRA number.
- **Retract**: the $n{=}250$ spine with per-arm filtered range, A2/A3/discrimination/TOST/MIA numbers, the Llama $n{=}24$ probe and its noisy $A0{=}0.88$ control, and the leakage-refusal-gate's E2 numbers (0% vs 100%, $n{=}15$ stratum).

## One judgment call flagged, not a numeric error

The boundary-law noise-floor sentence ("+0.022 ... below its own ~0.038 binomial noise floor") uses the mean of two per-condition DiD values (0.024, 0.020) rather than either individually, noted here so the source of that number is traceable.

## Not run

A fresh-reviewer context-appropriateness pass on every `\cite{}` (distinct from the numeric check above) is `paper/CITATION_AUDIT.md`'s job, already run separately on the flagged high-risk entries. A full per-citation context check against the now-finished paper text has not been run; recommended before any further revision, not required for this repository release.
