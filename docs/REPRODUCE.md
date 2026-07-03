# Reproduction Guide

Each stage is independent: different models, different GPU budgets, different (usually small) datasets. This doc gives the orchestration-level picture; **the authoritative, argparse-verified command for each script lives in that stage's own `code/<stage>/README.md`** — those were written by reading each script's actual CLI, not invented, and are what you should copy-paste from.

## General requirements

- A local **vLLM** server (OpenAI-compatible endpoint) for whichever model a given experiment needs — every script reads its endpoint from an environment variable or `--*-url`/`--*-gpu` flag, never hardcoded.
- Python 3.10+, one virtualenv per stage is simplest given differing dependency sets (`lightrag-hku`, `raptor-py`, `mem0`, vLLM itself).
- Datasets auto-download on first run: HoH (`datasets.load_dataset("russwest404/HoH-QAs")`), SciFact-Open (loaded from a local checkout — see `code/compile/README.md`).
- **GPU non-disruption**: several results in the paper (a dropped 2nd-model-scale run in `maintain/lossgate/`, a blocked static-audit probe in `compile/`) exist because these experiments were run on shared GPUs and deliberately did not contend with others' concurrent jobs. If you're on a shared box, check `nvidia-smi` before launching, and prefer the detached-run scripts (`detached_run.sh` in several stage dirs) which survive a crash without leaving orphaned processes.

## Compile (`code/compile/`)

- **7-arm HoH comparison**: needs a vLLM server for `Qwen2.5-14B-Instruct` (default `AGENT_URL=http://127.0.0.1:8101/v1`). Per-framework baselines (LightRAG, RAPTOR, mem0, Graphiti) need their own package installed (`pip install lightrag-hku`, etc. — exact packages are TODO-flagged in `code/compile/README.md` where the original setup used a source checkout instead of a package). Entry points: `real/pooled_hoh.py` (LLM-Wiki + closed-book + full-dump arms), `real/pooled_lightrag.py`, `real/pooled_raptor.py`, `real/run_mem0.py`, `real/run_graphiti.py`; resolver arm is `src/resolver.py`. Certificate/figures: `scripts/certify.py`, `scripts/fig_certificate.py`.
- **External validation** (`compile/external-validation/`): no local GPU required if using the Kilo API path (`LLM_API_KEY` + `LLM_BASE_URL`, see its README for the exact free-tier model IDs); otherwise `python wiki_decay.py --arm plain --model Qwen/Qwen3-8B --rounds 12 --seed 0` locally. This is the fastest thing in the repo to actually reproduce end-to-end — good first spot-check.

## Certify (`code/certify/`)

Needs a vLLM server for `Qwen2.5-14B-Instruct` (cross-family checks additionally need `Llama-3.1-8B-Instruct`), or set `CONTRACT_OFFLINE_MODEL` for an in-process vLLM engine instead of a separate server. Entry points: `contract/certify.py` (E1 gate), `contract/maintain.py` (E2/E2b maintained certificate + build lottery), `contract/repair.py` (E4 adaptivity trap), `contract/currency.py` + `contract/e5_prep.py` (E5 two-clause contract on HoH). CPU-only re-derivation scripts (`sizing.py`, `recompute_e1.py`, `recompute_e2b.py`) let you regenerate the certificate math from a saved results file without re-running any LLM calls.

## Maintain (`code/maintain/`)

- **Trained maintainer** (`maintain/trained-maintainer/`): needs a vLLM server for `Qwen2.5-14B-Instruct` with LoRA serving enabled for the trained arm. `sft_datagen.py` → `sft_train.py` produces the LoRA adapter; `p4_protocol.py` runs the tournament (vanilla/anchored/conservative/trained); `goodhart_eval.py` runs the boundary-law trap table.
- **LossGate** (`maintain/lossgate/`): same model family, typically two server ports (base + LoRA-enabled, e.g. `8102`/`8104`) since E4 gates around the trained-maintainer's adapter. Entry points: `run_a1.sh` (main make-or-break comparison), `run_e6.sh` (judge-free HoH replication), `run_e45.sh` (LoRA + τ-sweep arms).

## Retract (`code/retract/`)

- **agent-memory/** (primary result): needs a vLLM server for `Qwen3-14B` at `localhost:8102`, `mem0==0.1.118` + a local Chroma store. Main script is `b1_ladder.py` (configured via environment variables, not CLI flags — see its README for the exact variable list) — this is the $n{=}250$ A0–A4 ladder. `k_sweep.py` reproduces the 12-round native-consolidation trace. The Letta cross-substrate arm (`letta_ladder.py`) is optional and needs three additional services (a Letta server on `localhost:8285`, Postgres+pgvector, a CPU embedding shim on `localhost:8290`) — skip it unless you specifically want the directional second-substrate probe; the paper's primary claim does not depend on it.
- **leakage-refusal-gate/** (secondary result — read the framing caveat in its README before using anything but `e2_leakage.py`): needs two GPU-served models, `Qwen3-14B` (backbone) and `Qwen2.5-72B-Instruct` (judge), on separate GPUs (`--backbone-gpu`/`--judge-gpu`).

## What "reproduce" means here

Given the honesty theme of this paper, most stages report single-seed or few-seed results (see `docs/LIMITATIONS.md`) — re-running will not reproduce a number to the third decimal, and isn't meant to. What re-running should reproduce is the *qualitative* result and roughly the *scale* of the effect: resolution beats accumulation by an order of magnitude, the trained maintainer beats the best prompt by a small but significant margin, LossGate's coverage stays at or above its nominal target, and the membership gate reduces backflow surfacing to near zero while the correctness gate does not.
