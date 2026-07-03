# Limitations

This mirrors `paper/sections/07_limitations.tex` (the canonical version — read that for the numbers in context) with added notes for anyone trying to reproduce or extend this work.

## Compile

- HoH result is a single build, single run — no seed variance reported for the compile step itself. The build-lottery finding in the Certify stage (17–23% per-claim disagreement between identical builds) suggests this single-run limitation is not cosmetic; treat the compile-stage point estimates as one draw from a distribution, not the distribution's mean.
- Scaled substrate is HoH-only; the code-library substrate is a small consistency check ($n{=}14/13$), not a powered second benchmark.
- The static-audit cross-pipeline probe (SciFact-Open, different reader/embedder) could not be run as a matched comparison because of a shared-GPU thread-limit conflict with other jobs on the host at the time — infrastructure-blocked, not attempted-and-failed.
- The external-validation experiment (`code/compile/external-validation/`) is single-seed, same-model self-judged at temperature 0. If you re-run it with an independent calibrated judge, we'd genuinely like to know whether the numbers hold.

## Certify

- Model family is varied (Qwen vs. Llama) only on the fidelity gate experiment (E1c/E1d); every other certified number here uses Qwen for both compiler and judge role. Cross-family generality of the maintained-certificate and adaptivity-trap results specifically is untested.
- Judge-noise calibration uses a constructed present/absent pair set, not human-labeled ground truth.
- The dense-RAG comparator uses oracle page routing — it isolates page-content fidelity from retrieval-routing error, which flatters RAG relative to a real deployment where routing loss also applies.

## Maintain

- **The single largest open risk in this paper.** The reward-gaming boundary law's dangerous regime — an RL-trained maintainer, or one facing a fixed/predictable audit schedule, where gaming the audit signal is the direct training objective — has not been run. Everything reported here is the offline-selection regime, extrapolated to predict (not measure) the dangerous one.
- The binding-budget policy difference-in-differences (+0.022) sits below its own estimated noise floor (~0.038) — the ~6× signal-laundering ratio is directional, not precise.
- A second model scale was not run for LossGate, specifically to avoid contending for shared GPUs with others' concurrent jobs — disclosed, not hidden, and not the same as "attempted and failed."
- The LoRA-wrapped LossGate arm and the rollback-threshold (τ) sweep are each single-seed.

## Retract

- The primary $n{=}250$ ladder uses one composition template and one primary model (Qwen3-14B). The second-model probe (Llama) is $n{=}24$ with a noisy no-op control ($A0{=}0.88$ where $\approx\!1.0$ was expected) and tests a different memory architecture's block-plus-archival substrate, not a matched rerun of the same experiment. We describe this as "two substrates plus a directional probe," not "validated across two model families" — the $n{=}24$ evidence does not support the stronger framing.
- The complementary leakage-refusal-gate result has a small correct-and-retracted stratum ($n{=}15$) and the backbone and judge share a model family (circularity risk, disclosed here too).
- **Formally certifying retraction of a claim fused across multiple sources is an open problem this paper does not solve.** We explored a formal-criterion approach for this specific case and found it structurally unsound (the residue metric is blind to the fused-claim case it targets), so it is deliberately not carried into this paper's claims — see `code/retract/leakage-refusal-gate/README.md` for the full explanation and what's still usable from that codebase.

## Cross-cutting

- No stage in this repository has had a genuinely external (non-author) peer review pass. What shaped which results made it into the paper, and how they're scoped, is summarized throughout this document and the paper's own Limitations section rather than in a separate change log.
- Every headline experiment ran under a shared-GPU, non-disruptive-use policy; where that policy caused a run to be skipped or scaled down, we've tried to flag it at the specific result rather than only here.
