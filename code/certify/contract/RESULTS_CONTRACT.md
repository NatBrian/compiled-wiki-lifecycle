# Contract Experiments, Running Results Log

Pre-registered gates, tracked internally during development. All data external (SciFact-Open official).
Builder+judge: Qwen2.5-14B-Instruct, build temp 0.7 / judge temp 0.
Claim splits seed 0: CAL=40 (TPR), AUDIT=76, ABSENT=120 (FPR). Corpus N=500 (E1).

## ★ 2026-06-12, METHODOLOGY AUDIT + FIXES (3-agent audit: stats, data/leakage, baseline fairness)
Fixes applied and ALL affected experiments RE-RUN (offline in-process vLLM, batched, since the env blocked background GPU servers):
1. **Coverage validation was circular** (compared LCB to a self-derived point estimate). FIXED: replaced with simulation-from-known-R coverage (`stats.coverage_simulation`); union-bound LCB covers ≥0.999 ≥ nominal 0.95 (valid + conservative).
2. **FPR "absent" set contaminated**, 23/120 calibration claims had a CONTRADICT gold doc in corpus. FIXED: `split_claims` now drops any absent claim whose SUPPORT *or* CONTRADICT evidence is in the built corpus → 120 genuinely-absent.
3. **TPR pooled n=80 from 40 correlated claims** (anti-conservative). FIXED: single-gold n=40 distinct; buried-gold reported separately.
4. **E5 substring answer-matching** double-counted ~1% nested pairs. FIXED: nested-pair filter in `e5_prep`. Also E5 calibration now HELD OUT from audited probes (disjoint), and currency clause genuinely judge-noise-corrected (`stats.ser_upper`).
5. **Oracle-routing disclosure**, added to gate-table caption (wiki=routed content ceiling, RAG=end-to-end; conservative for the RAG-wins claim) + report RAG recall@5=0.961.

**Corrected headline numbers (all conclusions UNCHANGED, bounds more conservative):**
- E1 gate: wiki LCB **0.371** (R̂ 0.609), RAG LCB **0.740**; sim-coverage ≥0.999; TPR 0.800 (n=40), FPR 0.000 (120 clean), recall@5 0.961. → results_e1.json
- Cross-family 2×2 (refined cal): Qwen/Qwen 0.609/0.371; Qwen/Llama 0.576/0.343 (fpr_raw 0.05); Llama/Qwen 0.559/0.329. Corrected-R stable across families. → results_e1c/e1d.json
- E2b churn (recomputed `results_e2b_fixed.json`): stale cert 0.386 breached at **batch 7** (truth 0.378); maintained valid all batches. Lottery 17.1%/23.3% (calibration-independent, unchanged).
- E4 repair: adaptivity gap **0.52** (adaptive 0.760 vs honest 0.244); union honest 0.244→0.389. → results_e4.json
- E5 currency: born-stale (0/1.0) → current **R≥0.842, SER≤0.208** at 47% cost (504/1080), held-out calibration. → results_e5.json
- E2 cost ratio 40%, E3 lottery 23.3%, calibration-independent, unchanged.

`audit_numbers.py` 37/37 numbers trace to JSONs. NOTE old numbers in the dated sections below are PRE-FIX; the paper uses the corrected set above.

---

## E1, one-shot retention certificate (2026-06-11), GATE PASS
`certify.py` → `results_e1.json`. 535 calls, 2.6 min.

| arm | raw verdict rate | corrected R̂ | certificate LCB@95 | naive LCB | coverage (1000 subsamples) |
|---|---|---|---|---|---|
| wiki (63 pages, 8 docs/page, oracle page) | 0.434 | **0.589** | **0.347** | 0.337 | 1.000 |
| rag_dense (bge-small, k=5) | 0.803 | 1.000 | **0.813** | 0.712 | 1.000 |

Judge calibration (constructed ground truth from expert annotations):
- TPR single-gold 0.800, buried-gold 0.675 (pooled 0.7375, n=80), judge misses ~26% of present facts; context burying costs 12.5 pts.
- FPR 0.000 in BOTH styles (wiki-page n=120, raw n=120), judge never false-YESes absent claims; fpr_U@98.3% = 0.034.
- Gate: coverage 1.0 ≥ 0.93 ✓; TPR−FPR 0.74 > 0.3 ✓; R̂_wiki 0.589 ∈ (0.5,0.98) ✓; separation: RAG LCB 0.813 > wiki R̂ 0.589 ✓.

Reading: an *uncertified* store ships claiming nothing; a naive verdict-rate bound (0.337) is
accidentally valid here only because FPR=0; the corrected certificate is what licenses
"retention ≥ 0.347 with 95% confidence", and the certification PRICE (R̂−LCB ≈ 0.24 at n=76)
is itself a headline measurement: contract tightness is sample-limited, motivating audit-set
sizing and (E2) evidence reuse across updates.

## E1c, cross-family judge robustness (2026-06-11), #1 WEAKNESS CLOSED
`cross_judge.py` → `results_e1c.json`. Same Qwen-built E1 stores re-judged by Llama-3.1-8B-Instruct (Meta family, independent of Qwen compiler). 472 calls.

| | Qwen judge (E1) | Llama judge (E1c) |
|---|---|---|
| TPR | 0.738 | 0.775 |
| FPR wiki / raw | 0.000 / 0.000 | 0.000 / 0.058 |
| wiki corrected R̂ | 0.589 | **0.594** |
| wiki cert LCB | 0.347 | 0.364 |
| RAG cert LCB | 0.813 | 0.758 |

**Corrected retention is judge-family-INVARIANT** (0.589 vs 0.594), certificate measures a STORE property, not the judge. Validity holds (Llama TPR−FPR 0.72>0.3); wiki-vs-RAG separation preserved. Llama's nonzero raw-FPR (0.058) is exactly what the correction absorbs → demonstrates the correction's value. Refutes "self-judging artifact."

## E1d, cross-family COMPILER robustness (2026-06-12), last limitation closed
`build_store.py` (Llama-3.1-8B compiles the SciFact store) → `cross_judge.py` judges with Qwen (same judge as E1, isolates compiler). → `results_e1d.json`.

| | Qwen-built (E1) | Llama-built (E1d) | judge |
|---|---|---|---|
| wiki corrected R̂ | 0.589 | **0.624** | Qwen (both) |
| wiki cert LCB | 0.347 | **0.378** | Qwen |
| RAG cert LCB | 0.813 | 0.813 | Qwen (RAG = raw docs, compiler-independent) |

Llama-compiled store certifies comparably (R≥0.378), wiki-vs-RAG fidelity gap intact, ~38% compile loss reproduces (vs Qwen ~41%). Contract + separation + loss NOT compiler-specific. With E1c (judge swap) = full 2×2 family robustness.

## Sizing curve (2026-06-12), price of certification
`sizing.py` resamples E1 wiki audit verdicts at n=10..76 (no GPU). → `results_sizing.json`, `figs/sizing.pdf`.
Price (R̂−LCB) 0.457 (n=10) → 0.241 (n=76); LCB 0.132→0.348. Concretely sizes audit-vs-guarantee.

## E1b, TPR style-transfer diagnostic (2026-06-11), ASSUMPTION SAFE
`e1b_style.py` → `results_e1b.json`. Certificate direction requires TPR_raw ≥ TPR_compiled-style.
1-doc-compiled pages (fact present a.s.): judge rate 0.75 vs raw 0.80 → gap ≤ noise (n=40), and
0.75 itself lower-bounds style-TPR (contains 1-doc compile loss). One-directional assumption holds.

## E2/E3, incremental maintenance + lottery transfer (2026-06-11)
`maintain.py` n_docs0=400→50 pages, 8 batches (~16 docs/batch, 16 held-back audit golds injected 2/batch). 1092 calls, 14.3 min. → `results_e2e3.json`.

**E3 build lottery:** two clean builds of identical corpus disagree on **23.3%** of per-claim verdicts. Build-A certificate LCB 0.327; build-B corrected truth 0.506 → transfer happens to stay valid HERE only because the certificate is conservative enough to absorb cross-build variance, but per-FACT guarantees do not transfer (23% flip). ⇒ certificate is a per-artifact, store-level object; rebuilding re-rolls which facts are present.

**E2 maintained vs full (realistic mixed stream):** STALE/FULL/MAINTAINED all stay valid (LCB ≤ corrected truth ~0.50) across 8 batches; MAINTAINED matches FULL validity at **40.5% of audit cost** (248 vs 612 calls). Stale stayed valid here because injecting fresh correctly-compiled facts offset incremental rewrite damage → store-level retention healthy (~0.50 throughout). Shows maintenance is CHEAP+SUFFICIENT when the store is healthy. (Necessity shown by E2b churn.)

## E2b, adversarial churn (drift isolation) (2026-06-11), NECESSITY SHOWN
`maintain.py --n_hold_gold 0 --batch_fillers 40` (pure rewrite pressure on fixed 76-probe set, no new facts). 31 min. → `results_e2b.json`.

| batch | 1–3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| corrected truth R | 0.633 | 0.560 | 0.579 | 0.488 | **0.398** | **0.398** |
| stale cert LCB (t=0) | 0.424 | 0.424 | 0.424 | 0.424 | 0.424 | 0.424 |
| stale valid? | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** |
| maintained LCB | 0.393 | 0.332 | 0.347 | 0.287 | 0.201 | 0.201 |
| maintained valid? | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Robust for 3 batches (drift negligible at low rewrite count) then telephone effect compounds → truth falls monotone 0.633→0.398. **STALE certificate BREACHED at batch 7** (claims R≥0.424, true R=0.398, overclaim on live store). MAINTAINED tracks down, valid every batch. Cost 581/684 = 85% (heavy churn shrinks saving but maintenance = only valid non-full policy). NECESSITY half of E2. (E3 lottery here 17.1%, corpus 200.)

## E5, unified two-clause contract on HoH supersession stream (2026-06-11, JUDGE-NOISE-CORRECTED)
`e5_prep.py` (HoH-QAs, 160 currency probes + 240 distractors) + `currency.py` (8 batches). Both clauses judge-noise-corrected: calib ret_tpr 42/60, ret_fpr 3/60, ser_tpr 44/60, ser_fpr 2/60. → `results_e5.json`. (Old uncorrected run archived `results_e5_uncorrected.json`, do NOT cite.)

| batch | superseded | retention LCB | SER upper bound | maint/full cost |
|---|---|---|---|---|
| 0 (all stale) | 0 | 0.000 | 1.000 | n/a |
| 2 | 40 | 0.000 | 0.969 | 304/480 |
| 4 | 80 | 0.235 | 0.668 | 448/800 |
| 6 | 120 | 0.451 | 0.405 | 605/1120 |
| 8 (all current) | 160 | **0.756** | **0.075** | **749/1440** |

Store born stale (R 0, SER_ub 1.0, certificate honestly refuses to bound currency while store asserts stale facts) → maintained contract certifies full transition to R≥0.756, SER≤0.075 at 52% audit cost; maintained == full validity at every batch. Fidelity+currency certified jointly+continuously on one evolving store. (Note: corrected retention LCB can exceed the raw verdict rate because the TPR<1 correction lifts the bound, expected, valid.)

## E4, certified repair: the adaptivity trap (2026-06-11)
`repair.py` N=400, repair-set 38 / holdout 38 disjoint. 473 calls, 2.7 min. → `results_e4.json`.

| strategy | repair-set cert (LCB) | holdout cert (LCB) | pages rebuilt | note |
|---|---|---|---|---|
| initial | 0.244 | 0.300 | n/a | baseline |
| **targeted** (test-guided) | **0.686** (r̂=1.0) | **0.217** (r̂=0.50) | 19 | **adaptive overclaim = +0.469**; true retention DROPPED 0.61→0.50 (lottery re-roll damages other pages) |
| full rebuild | 0.272 | 0.300 | 50 | no real gain (lottery) |
| compile-twice-union | 0.329 | **0.389** | 50 | genuine +0.09 honest gain, 2× store cost |

**Headline:** a WiCER-style repair loop certified on its OWN diagnostic probes overclaims retention by **47 points** and can silently *worsen* the store; only certification on untouched holdout probes is valid. Compile-twice-union is the repair that honestly raises the certified bound.

## Citation verification (fetch-confirmed 2026-06-11)
- **2509.20461** Kuwahara, "Document Summarization with Conformal Importance Guarantees," NeurIPS 2025, single-doc, extractive, one-shot, NO store/updates/supersession. (closest prior; delta = persistent corpus store + maintenance + currency.)
- **2606.09877** Huerta, "Streaming Knowledge Compilation: Proactive Materiality-Scored Pinning for Time-Evolving LLM Wikis," Jun 3 2026, O(√T log K) regret, NO retention/supersession guarantee. (scoop clock; same setting, no certificate.)
- **2605.07068** Huerta, "WiCER: Wiki-memory Compile, Evaluate, Refine," May 8 2026, heuristic test-guided repair, NO certificate (empirical 80% recovery). (our E4 shows why its self-probe certification is invalid.)
- **2601.20913** Chen Feng, "Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges," Jan 28 2026, TPR/FPR-calibrated finite-sample-valid threshold. (STATISTICAL FOUNDATION our certificate builds on; we extend to persistent compiled store + incremental maintenance + currency.)
- Both Huerta papers = same author owns heuristic-repair + regret-bound compiled-wiki space; NEITHER certifies retention/currency → our lane.
