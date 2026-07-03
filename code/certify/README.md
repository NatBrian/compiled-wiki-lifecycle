# certify/ — the (δ, ε) compilation contract

Code for certifying a compiled LLM-wiki store against two clauses:

- **fidelity**: retention `R ≥ 1 − δ` — the compiled store still supports the facts
  it is supposed to support.
- **currency**: staleness-error-rate `SER ≤ ε` — the compiled store does not assert
  facts that have since been superseded.

Both clauses are issued as **finite-sample lower/upper confidence bounds** (Clopper–Pearson,
via `stats.py`), with an explicit **judge-noise correction**: an LLM judge used to verify
claims against store text has its own true-positive/false-positive rate (measured by
calibration probes), and the raw judge-verdict rate is corrected through that TPR/FPR before
the confidence bound is computed (`stats.corrected_point`, `stats.retention_lcb`,
`stats.ser_upper`). A naive bound computed directly on raw judge verdicts is optimistic; the
corrected bound is what is actually defensible as a "certificate."

## Layout

```
certify/
  contract/     experiment scripts (E1, E2/E3, E2b, E4, E5) + the stats/client/store
                library that this stage's own experiments and the maintain/ stage both use
  harness/      dataset loaders (SciFact-Open, HoH-QAs) + earlier MOAT/adversarial-probe
                exploration scripts kept for provenance
```

The library modules shared with the maintain/ stage elsewhere in this codebase
(`code/shared/{oai_client,maintain,certify,stats,currency}.py`, `code/shared/data.py`)
are canonical copies of `contract/{oai_client,maintain,certify,stats,currency}.py` and
`harness/data.py` respectively — see `code/shared/README.md` for the exact
file → importer mapping.

## Requirements

- Python 3.11, `vllm` (only needed for the `CONTRACT_OFFLINE_MODEL` in-process path).
- A running local chat-completions endpoint for the builder/judge model, OR the
  `CONTRACT_OFFLINE_MODEL` environment variable set to load the model in-process via
  vLLM's offline `LLM` engine (see `contract/oai_client.py`):
  - `VLLM(host="http://127.0.0.1:8102", model="qwen2.5-14b")` is the default factory
    used throughout — point a local vLLM server serving **Qwen2.5-14B-Instruct** at
    `:8102` (builder + judge for E1/E2/E3/E2b/E4/E5).
  - Cross-family robustness checks (`cross_judge.py`, `build_store.py`) additionally need
    a second server for **Llama-3.1-8B-Instruct**.
  - For offline/in-process use (no server): `export CONTRACT_OFFLINE_MODEL=<hf-model-id>`
    (optionally `CONTRACT_GPU_UTIL`, `CONTRACT_MAXLEN`); note the offline engine serializes
    calls through one lock, so pass `--workers 1` in that mode.
- Datasets (external, not redistributed here):
  - **SciFact-Open** — `corpus_candidates.jsonl` + `claims.jsonl` in the format documented
    at the top of `harness/data.py`; default path expected at
    `<repo>/benchmarks/scifact-open/data` (override with `--data`).
  - **HoH-QAs** (`russwest404/HoH-QAs`) — used by `contract/e5_prep.py` to build the
    currency/supersession substrate (`e5_data.json`, already included as a small example).

## E1 — validity gate (one-shot retention certificate)

```
python contract/certify.py \
    --data <path to scifact-open>/data --n_docs 500 --docs_per_page 8 \
    --n_cal 40 --n_fpr 120 --k_rag 5 --alpha 0.05 --n_splits 1000 \
    --out results_e1.json --store_out store_e1.json
```

Builds a compiled wiki store (WIKI arm) and a dense-RAG baseline (RAG arm) over a
SciFact-Open corpus, judges audit claims against each, corrects for judge TPR/FPR, and
reports the corrected retention certificate LCB for both arms plus empirical coverage
(via `stats.coverage_simulation`) over resampled audit subsamples.

Gate (from `RESULTS_CONTRACT.md`): simulation coverage ≥ 0.93, TPR − FPR > 0.3,
R̂_wiki ∈ (0.5, 0.98), and RAG LCB > wiki R̂ (separation). Reported result:
wiki LCB **0.347–0.371** (build/judge-family dependent) vs RAG LCB **0.740–0.813**.

Diagnostics that reuse `certify.py`'s TPR-direction assumption / robustness:
- `python contract/e1b_style.py` — style-transfer check that TPR_raw ≥ TPR_compiled-style.
- `python contract/cross_judge.py` — re-judges an existing store with a different judge
  family (Llama) to rule out self-judging artifacts (E1c).
- `python contract/build_store.py` — compiles the store with a different builder family
  (Llama) to rule out compiler-specific effects (E1d), then re-judge with `cross_judge.py`.
- `python contract/sizing.py` — resamples E1's audit verdicts at n=10..76 (no GPU needed)
  to show the "price of certification" (R̂ − LCB) as a function of audit-set size.

## E2 / E2b — maintained certificate under churn

```
python contract/maintain.py \
    --data <path to scifact-open>/data --n_docs0 240 --docs_per_page 8 \
    --n_cal 40 --n_fpr 120 --n_hold_gold 16 --batches 8 --batch_fillers 24 \
    --refresh_k 8 --alpha 0.05 --cal_file results_e1.json --out results_e2e3.json
```

Streams updates into an existing store over `--batches` rounds and compares three
certificate-maintenance policies at every step: `STALE` (never re-audited), `FULL`
(re-audit every in-store probe every batch), and `MAINTAINED` (re-judge only probes on
pages that changed this batch, plus a `--refresh_k`-sized random refresh sample; unchanged
pages carry over their prior verdict since the judge is temperature-0). Also runs E3: a
second independent build B is judged against the same probes to measure "build lottery"
transfer miscoverage. Result: MAINTAINED matches FULL's validity at ~40% of audit cost;
build-to-build disagreement (lottery) ≈ 17–23%.

**E2b (adversarial churn / necessity)** — same script, pure-rewrite-pressure config
(no new certified facts injected, only filler churn on a fixed probe set):

```
python contract/maintain.py --n_hold_gold 0 --batch_fillers 40 --out results_e2b.json
```

Then recompute both E2b's and E1's certificates under the refined (post-audit) judge
calibration without re-running any LLM calls (CPU only):

```
python contract/recompute_e1.py       # -> results_e1_fixed.json
python contract/recompute_e2b.py      # -> results_e2b_fixed.json
```

Result: the `STALE` certificate (issued at t=0) is **breached at batch 7** (claims
R ≥ 0.424, true R = 0.398) while `MAINTAINED` stays valid every batch — this is the
necessity argument for maintenance (E2 alone only shows it's cheap+sufficient on a
healthy store; E2b shows a static certificate can silently go wrong).

## E4 — certified repair / the adaptivity trap

```
python contract/repair.py \
    --data <path to scifact-open>/data --n_docs 400 --docs_per_page 8 --n_cal 40 \
    --alpha 0.05 --cal_file results_e1.json --out results_e4.json
```

A store that fails its contract (LCB < R_min) is repaired three ways (`targeted` /
`rebuild` / `union`), then re-certified two ways: **adaptive** (re-audit the same
repair-guided probes — invalid, inflated) vs **honest** (audit a disjoint holdout
set — valid). Demonstrates the overclaim a test-guided repair loop (WiCER-style) would
ship if it certified on the same probes it repaired against: adaptive LCB 0.686 vs
honest LCB 0.217 on the same repaired store (gap ≈ 0.47), while true retention actually
*dropped*. `union` (compile twice, store both variants) is the only strategy with a
genuine (non-adaptive) gain, at 2× store cost.

## E5 — currency certificate on HoH (two-clause contract under supersession)

```
python contract/e5_prep.py --n_probes 160 --n_distractors 240 --out e5_data.json
python contract/currency.py \
    --data e5_data.json --docs_per_page 8 --batches 8 --n_audit 120 \
    --refresh_k 10 --alpha 0.05 --out results_e5.json
```

`e5_prep.py` turns HoH-QAs into an evolving corpus: the store is born compiled from
*stale* evidence only (retention 0 / SER upper bound 1.0 by construction), then a
supersession stream replaces stale evidence with current evidence in batches.
`currency.py` maintains **both** clauses jointly on the same evolving store — retention
(current answer verifiable) and SER (stale answer no longer assertable) — using the
judge-noise-corrected `stats.ser_upper` / `stats.retention_lcb`, with maintained-vs-full
validity checked at every batch. Result: store moves from R=0.0/SER≤1.0 to
R≥0.756/SER≤0.075 by the final batch, at ~52% of full-audit cost.

## Other scripts in `contract/`

- `audit_numbers.py` — traces every number quoted in the paper back to a results JSON
  (integrity check, no GPU).
- `make_figs.py` — regenerates the paper figures from the results JSONs.
- `driver_qwen1.py` — thin driver wrapper used for one specific run configuration.

## `harness/`

Dataset loaders (`data.py`, imported by every `contract/` script) plus an earlier line of
exploration (`run_moat*.py`, `adversarial.py`, `scan.py`, `methods.py`, `client.py`,
`run_scifact_absence.py`, `run_floor_pilot.py`) kept for provenance; these are not on the
critical path for E1/E2/E2b/E4/E5 above but `data.py` is (and is the file re-exported as
`code/shared/data.py`).
