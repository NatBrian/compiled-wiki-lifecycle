# Agent-Memory Backflow: A Membership Gate for Durable Retraction

This is the strongest validated result in this repository. It shows that
**deleting a fact from an AI agent's memory does not make it stay deleted** if
the agent has its own autonomous memory-consolidation loop, and it shows a
simple fix (a *membership* write-veto, not a *correctness* filter) that
closes the gap essentially completely.

## The problem: "backflow"

Modern agent-memory frameworks (Mem0, Letta, and similar) don't just store
verbatim chunks of what a user told them. They periodically **re-summarize**
recent conversation into durable memory notes, a "consolidation" pass. If
you retract a fact (delete it from the agent's memory store), but the
surrounding conversation transcript that *implies* the fact is still present
(or gets mentioned again, even obliquely), the next consolidation pass can
**re-derive and re-write the retracted fact right back into memory**. The
deletion didn't stick, it just got silently undone by the agent's own
write-back loop.

We call this **backflow**: a synthesized memory fact that was deleted
resurfaces because the agent's consolidation mechanism reconstructs it from
surviving evidence, not because deletion failed to run.

The core finding of this codebase is that **the obvious fix doesn't work**.
If your write-gate on the consolidation loop only checks "is this memory
entry *faithful/correct* w.r.t. the transcript?" (a natural thing to check,
you don't want the agent hallucinating), a faithful re-derivation of the
retracted secret **passes the correctness gate** and gets re-written. Staying
correct is not the same thing as staying deleted. The fix that actually works
is a **membership veto**: at write time, explicitly check whether the
candidate memory text reconstructs the value you retracted, and drop it if
so, regardless of whether it's "correct."

## The A0–A4 arm ladder

`b1_ladder.py` is the main experiment. For each seeded fact, it runs an
identical lifecycle (ingest → retract from the synthesized store → N rounds
of autonomous consolidation → probe) under five arms that differ only in the
consolidation **write hook**:

| Arm | Write hook | Expected outcome |
|-----|-----------|-------------------|
| **A0** | No retraction (control) | Fact stays, Resurfacing Success Rate (RSR) ≈ 1.0, sanity baseline |
| **A1** | Naive delete, no write gate | Backflow, RSR high |
| **A2** | Delete + correctness/faithfulness write-gate | Backflow **persists**, a faithful re-derivation of the secret is *allowed* by a correctness check, so RSR stays high. This is the key negative result: correctness ≠ deletion. |
| **A3** | Delete + **membership write-veto** (the fix) | RSR ≈ 0, anything reconstructing the retracted value is vetoed at write time regardless of correctness |
| **A4** | Never-ingested oracle | RSR ≈ 0, the secret was never told to the agent at all; this is the counterfactual floor, not a "fix," just a sanity ceiling for how low RSR can go |

Each run also probes a **benign, non-secret preference** that is never
retracted (`benign_retained`), to check that the membership veto in A3 isn't
just nuking memory indiscriminately (over-veto / collateral forgetting).

**Discrimination** is defined as `RSR(A2) - RSR(A3)`: how much better the
membership gate does than the correctness gate at the *same* task.

## Headline numbers (n=250 spine, `results/b1_ladder.json`)

- **A2 (correctness gate) RSR = 0.990** (Wilson 95% CI [0.966, 0.997]), the
  correctness gate lets the secret leak back into memory on almost every trial.
- **A3 (membership veto) RSR = 0.000** (Wilson 95% CI [0.000, 0.020]), the
  membership gate blocks it essentially completely.
- **Discrimination (A2 − A3) = 0.990**, membership, not correctness, is
  what actually stops backflow.
- **benign_retained = 1.0 across all arms**, the membership veto does not
  cause collateral over-deletion of unrelated (non-retracted) memory.
- **Membership-inference-attack (MIA) cross-check** (`mia.py`, Min-K%/Min-K%++
  statistics over Qwen3-14B logprobs): A1 (leaked/ungated) memory is
  MIA-distinguishable from the never-ingested oracle A4 (**AUC ≈ 0.99**),
  while A3 (gated) memory is statistically **indistinguishable** from A4
  (**AUC ≈ 0.50**). This is an independent, judge-free confirmation that the
  membership-gated store really does look like the secret was never ingested,
  not just that a downstream QA probe fails to surface it.

Reference summary JSONs (all well under 500KB) are checked in under
[`example_results/`](example_results/) as a concrete example of the output
shape, `b1_ladder.json` is the full n=250 run backing the numbers above;
`mia.json`, `cost.json`, `recon.json`, `sanity.py`'s `sanity.json`, and
`stats_cluster.json` are the companion analyses described below.

## File map

Core experiment scripts:
- `e0a_ignorance.py`, sanity check that the backbone has no parametric
  knowledge of the seeded private facts before any memory is involved.
- `e0b_backflow.py`, `e0b_inference.py`, single-consolidation-pass backflow
  and inference-leakage probes (precursors to the full ladder).
- `e0b_auto.py`, `e0b_auto_inf.py`, autonomous multi-round consolidation
  loop versions (A1/A3/A4 only) that `b1_ladder.py` extends into the full
  A0–A4 ladder.
- `b1_ladder.py`, **the main experiment.** See below for how to run it.
- `run_b1_llama.sh`, re-runs the ladder on a second, cross-family backbone
  (Llama-3.1-8B) for generality.
- `b2_ablation.py`, ablates pieces of the membership-gate mechanism.
- `b2b_necessity.py`, necessity check (is the gate doing anything a simpler
  mechanism couldn't).
- `b4_attack.py`, adversarial probing of the gate (paraphrase/indirect
  attacks trying to sneak the secret back in).
- `letta_ladder.py`, the same A-arm ladder replicated on a **second, fully
  independent agent-memory substrate** (Letta) instead of Mem0, to show the
  membership-gate result is not an artifact of one framework. **Optional /
  secondary**, see setup notes below.
- `k_sweep.py`, sweeps number-of-consolidation-rounds to show RSR/round
  curves (does backflow accumulate over more rounds, or plateau).
- `mia.py`, the Min-K%/Min-K%++ membership-inference-attack cross-check
  described above.
- `cost.py`, measures the per-hook overhead ("tax") of the write gate.
- `recon.py`, checks the tombstone/deletion record itself doesn't leak the
  retracted value (non-invertibility of the deletion marker).
- `sanity.py`, gold-set sanity check of the gate predicate itself (does the
  membership-veto's own judgment call match a small hand-labeled gold set).

Merge / orchestration:
- `merge_auto.py`, `merge_auto_inf.py`, `merge_b1.py`, `merge_b2.py`,
  `merge_b2b.py`, `merge_b4.py`, `merge_ksweep.py`, merge chunked partial
  result files (see chunking note below) into one final summary JSON.
- `run_b1_chunked.sh`, `run_b2_chunked.sh`, `run_b2b_chunked.sh`,
  `run_b4_chunked.sh`, chunked drivers for the corresponding experiments.
- `run_scale_spine.sh`, the actual driver used to produce the paper's n=250
  spine (`b1_ladder.py` + `e0b_auto.py`, chunked).
- `run_remaining.sh`, runs the lighter-weight follow-on analyses
  (`stats_cluster.py`, `sanity.py`, `cost.py`, `recon.py`, `mia.py`,
  `k_sweep.py`) after the n=250 scale run finishes.

Shared support modules (not experiments themselves, but required by every
script above, included so the repo is actually runnable, not just readable):
- `llm.py`, thin OpenAI-compatible chat client wrapper for the local vLLM
  server, including Qwen3 `/no_think` handling.
- `facts.py`, synthetic private-fact generator (entities, attributes,
  values, conversational contexts) used to seed each trial. Facts are
  generated programmatically at run time, no external dataset file is
  needed.
- `mem0_backend.py`, wires Mem0 0.1.118 to the local vLLM backbone + Chroma
  vector store, and provides the `contains_value` / `memories_with_value`
  helpers the write hooks use to check membership.
- `embed_server.py`, a tiny CPU-only OpenAI-compatible `/v1/embeddings`
  shim (BAAI/bge-small-en-v1.5), needed only for the **Letta** replication
  (Letta wants a real embeddings endpoint; the main Mem0 path does its own
  embedding internally).
- `stats_cluster.py`, re-runs GEE/TOST-style clustered statistics over the
  n=250 spine (accounts for repeated measures per fact/arm).

## Why chunked shell wrappers exist

`b1_ladder.py` and friends run through Mem0/Chroma, which leaks non-daemon
threads that eventually hang process shutdown under sustained load. The
`run_*_chunked.sh` / `run_scale_spine.sh` wrappers work around this by
invoking the Python script repeatedly as **fresh, short-lived processes**
(`P7_PART=1`, writing per-chunk part files), each covering a small offset
range of the fact set, then calling the matching `merge_*.py` to stitch the
part files back into one summary. This is an operational workaround, not
part of the method, if you're re-running at small `n`, you can call
`b1_ladder.py` directly (see below) and skip the chunking machinery.

## How to run the main result

`b1_ladder.py` takes **no CLI flags**, every knob is an environment
variable (reflecting the actual `os.environ.get(...)` calls in the script,
not invented flags):

| Env var | Default | Meaning |
|---|---|---|
| `P7_N_B1` | `40` | number of facts to run |
| `P7_OFFSET` | `0` | starting offset into the fact set (for chunking) |
| `P7_WORKERS_B1` | `2` | thread-pool worker count |
| `P7_N_CONSOLID` | `2` | number of autonomous consolidation rounds per trial |
| `P7_CHROMA_B1` | `/tmp/p7_b1` | scratch root for the per-thread Chroma stores |
| `P7_PART` | unset | if set, write a per-chunk `results/b1{TAG}_part_{OFFSET:03d}.json` instead of the merged summary |
| `P7_TAG` | `""` | suffix tag for output filenames (e.g. `_llama` for the cross-backbone run) |
| `P7_VLLM_URL` (via `llm.py`) | `http://localhost:8102/v1` | vLLM OpenAI-compatible endpoint |
| `P7_MODEL` (via `llm.py`) | `Qwen/Qwen3-14B` | backbone model name |

A small smoke run:

```bash
cd code/retract/agent-memory
P7_N_B1=10 P7_N_CONSOLID=2 P7_WORKERS_B1=2 python b1_ladder.py
# The script computes ROOT as two directories up from its own location
# (os.path.dirname(os.path.dirname(__file__))), so from this path that's
# code/retract/, output lands at code/retract/results/b1_ladder.json, and
# it looks for (optional) seed facts at code/retract/data/facts.json,
# falling back to facts.py's programmatic generator if that file is absent.
```

To reproduce the full n=250 spine as originally run:

```bash
CHUNK=10 TOTAL=250 bash run_scale_spine.sh
```

(Adjust the hardcoded `cd` path at the top of `run_scale_spine.sh` /
`run_b1_llama.sh` to point at wherever you've cloned this repo.)

### Requirements to run the main result (Mem0 path)

- A **local vLLM server** exposing an OpenAI-compatible chat endpoint for
  **Qwen3-14B** at `http://localhost:8102/v1` (override via `P7_VLLM_URL`).
- **`mem0` version `0.1.118`** specifically (pinned, later versions changed
  the consolidation/native-memory mechanism the experiment relies on) with a
  **Chroma** vector store backend.
- Python packages: `openai`, `mem0` (0.1.118), `chromadb` (pulled in by
  mem0), `torch` (mem0_backend imports it), `numpy` (for `mia.py`/`recon.py`).

That's it, this is the whole setup needed to reproduce the headline A0–A4
numbers above.

### Optional / secondary: cross-substrate replication on Letta

`letta_ladder.py` reruns the same A0–A4 ladder logic on **Letta** instead of
Mem0, to show the membership-gate result generalizes across agent-memory
frameworks. **This is a secondary, cross-substrate replication, not required
to reproduce the main result above.** It needs substantially more
infrastructure:

- A running **Letta server** at `http://localhost:8285` (override via
  `LETTA_BASE_URL`).
- **Postgres with the `pgvector` extension** (Letta's storage backend).
- A **CPU embedding shim** at `http://localhost:8290/v1` (override via
  `P7_EMB_URL`), run `python embed_server.py` for this; it serves
  BAAI/bge-small-en-v1.5 on CPU in OpenAI embeddings wire format since Letta
  needs a real embeddings endpoint (Mem0 doesn't, for the main path).
- The same vLLM Qwen3-14B server as above (`P7_VLLM_URL`, default
  `http://localhost:8102/v1`).

If you only care about the headline backflow/membership-gate result, you can
safely ignore `letta_ladder.py` and everything in this section, don't set
up Letta/Postgres/pgvector just to reproduce the main claim.
