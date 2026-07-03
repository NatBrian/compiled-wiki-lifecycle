# Serving-Time Leakage-Refusal Gate

This code implements a serving-time gate that refuses to leak retracted facts even when the backbone model answers correctly from parametric memory (E2 script: `e2_leakage.py`). This result — 0% leakage with the gate vs 100% without it, on correct-and-retracted queries — is the ONLY claim from this codebase used in the paper. The other scripts here (`e0_pilot.py` fusion-existence pilot, `e1_oracle.py` oracle-matched partial deletion, `e3_pareto.py` compression/severability sweep, `e4_scale.py` incremental-cost scaling, `e5_ablations.py`) implement a broader 'formal criterion for deleting fused claims' that we found to be circular / blind to the exact fused-claim case it targets (the residue metric can't detect marginal-content leakage from a source that contributed to a claim jointly with other sources — see the paper's Limitations section for the full explanation). They are included here for research transparency and because e1/e3/e4/e5 may still be useful building blocks for whoever tackles that open problem, but their headline numbers (e.g. 'zero residue on fused claims') should NOT be cited as validated — only e2_leakage.py's result is.

## What actually validates (`e2_leakage.py`)

E2 tests a simple but sharp distinction: **leakage is not the same thing as
error.** A backbone model can answer a query *correctly* purely from its own
pretraining (parametric memory) even after the source that would have
licensed that answer has been retracted from the compiled knowledge base.
Standard correctness-risk certification (e.g. C-RAG-style) only checks
whether the answer is right — it structurally cannot catch this case,
because the answer *is* right, it's just no longer supposed to be
sayable.

E2 measures, in the **correct-and-compiled stratum** (queries where the
backbone answers correctly *and* the (pre-retraction) compiled knowledge
base surfaced the same answer):
- **RAG leak rate** — a standard retrieval-augmented baseline
- **Param-only leak rate** — backbone alone, no retrieval
- **CRAG-approved-but-should-refuse rate** — fraction of RAG's leaks that a
  correctness certificate would have approved anyway (i.e., the leaks a
  correctness-only cert is blind to)
- **Our (gated) leak rate** — the serving-time gate refuses to answer unless
  the current (post-retraction) compiled knowledge base still supports the
  claim via a live source-provenance check, independent of whether the
  answer happens to be factually correct.

The measured result was **0.0% leak rate with the gate vs 100.0% leak rate
for RAG**, on the correct-and-compiled stratum, with the correctness-certificate
baseline missing roughly 73% of those leaks. That is the one number from this
codebase that made it into the paper.

## What does NOT validate — read this before citing anything else here

`e0_pilot.py`, `e1_oracle.py`, `e3_pareto.py`, `e4_scale.py`, and
`e5_ablations.py` build out a more ambitious claim: a general formal
criterion for *severing* (deleting) a claim that was fused from multiple
sources, with a "residue" metric meant to certify the deletion left no
detectable trace. We found this criterion **circular and blind to the exact
case it was built for**: the residue metric cannot detect marginal-content
leakage from a source that contributed to a claim *jointly* with other
surviving sources, because the metric only checks whether the claim as a
whole is still derivable — not whether a specific source's marginal
contribution survived inside a claim that other sources can also still
support. See the paper's Limitations section for the full explanation of
why this is a structural gap, not a tuning issue.

These five scripts are kept in this directory for **research transparency**
(so the full history of what was tried is auditable) and because pieces of
`e1_oracle.py` (oracle-matched partial deletion), `e3_pareto.py`
(compression/severability tradeoff sweep), `e4_scale.py` (incremental
maintenance cost scaling), and `e5_ablations.py` may still be useful
building blocks for whoever picks up the open problem of a *correct*
fused-claim deletion criterion. **Do not cite their headline numbers (e.g.
"zero residue on fused claims") as validated results** — only
`e2_leakage.py`'s leakage-refusal number above is.

## File map

- `e2_leakage.py` — **the validated result.** Serving-time leakage-refusal
  gate vs RAG/param-only/CRAG baselines on the correct-and-compiled stratum.
- `e0_pilot.py` — fusion-existence pilot (do multiple sources actually get
  fused into single claims in practice). NOT independently validated as a
  deletion criterion — see caveat above.
- `e1_oracle.py` — oracle-matched partial-deletion baseline (what would an
  omniscient deleter that knew exactly which claims to touch do). Feeds the
  fused-claim severability line of work; same caveat applies.
- `e3_pareto.py` — compression/severability Pareto sweep. Same caveat.
- `e4_scale.py` — incremental-maintenance cost scaling as the knowledge base
  grows. Same caveat.
- `e5_ablations.py` — ablations over the fused-claim deletion criterion.
  Same caveat.
- `p2_decomposition.py` — decomposition analysis used alongside the above;
  same caveat (not part of the validated E2 leakage-refusal claim).

## Requirements to run E2 (`e2_leakage.py`)

Two separate local vLLM OpenAI-compatible servers are required, on
different GPUs (exact flags below are pulled directly from the script's
`argparse`, not invented):

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `cpu` | `cpu`: build the query set only, no GPU calls. `gpu`: run the full experiment (needs both servers below). |
| `--backbone-gpu` | `4` | GPU index for the **backbone** vLLM server. Port = `8000 + GPU_IDX` (default → port 8004). |
| `--judge-gpu` | `6` | GPU index for the **judge** vLLM server. Port = `8000 + GPU_IDX` (default → port 8006). |
| `--backbone-model` | `Qwen/Qwen3-14B` | Model served by the backbone vLLM instance. |
| `--judge-model` | `Qwen/Qwen2.5-72B-Instruct` | Model served by the judge vLLM instance (fallback: `Qwen/Qwen3-14B` if 72B isn't available). |
| `--e1-results` | `<results_dir>/e1_results.json` | Path to E1 results (retraction candidates + wiki provenance) — only needed if you're chaining off an E1 run. |
| `--wiki-path` | `<repo_root>/data/wiki/compiled_wiki.json` | Path to the compiled wiki JSON with provenance annotations. |
| `--rwku-path` | `<repo_root>/data/rwku/rwku_full.jsonl` | RWKU dataset (NeurIPS 2024, jinzhuoran/RWKU). |
| `--muse-path` | `<repo_root>/data/muse/muse_full.jsonl` | MUSE dataset (ICLR 2025). |
| `--claims-per-source` | `15` | Atomic claims extracted per source document. |
| `--max-rwku-per-entity` | `10` | Max RWKU probes per entity. |
| `--max-muse-per-entity` | `10` | Max MUSE probes per entity. |
| `--seed` | `42` | Random seed. |

Concretely: bring up vLLM serving **Qwen3-14B** on one GPU and
**Qwen2.5-72B-Instruct** (the independent judge — deliberately a different,
larger model than the backbone, to avoid the backbone judging itself) on a
second GPU, at ports `8000 + backbone_gpu` and `8000 + judge_gpu`
respectively, then run:

```bash
python e2_leakage.py --mode gpu --backbone-gpu <idx> --judge-gpu <idx> \
  --wiki-path /path/to/compiled_wiki.json \
  --rwku-path /path/to/rwku_full.jsonl \
  --muse-path /path/to/muse_full.jsonl
```

`--mode cpu` (the default) only assembles the query set and does not require
either GPU server — useful for checking the harness runs before committing
GPU time.
