# External validation: SciFact-Open + a real third-party LLM-wiki tool

This is an independent replication of the headline "iterated rewriting loses
information" finding, on a **different dataset** (SciFact-Open, real PubMed
abstracts, not HoH) using a **different tool's own update policy** (a real
Hermes-style LLM-wiki SKILL.md, not our benchmark harness's prompt).

The main `code/compile/` experiment shows that curated, overwrite-on-ingest
wiki pages beat accumulating baselines on synthetic-timeline HoH questions.
The question this experiment answers is narrower and orthogonal: **does the
compiled wiki itself degrade if you let it repeatedly rewrite its own pages**,
using the update policy that a real published LLM-wiki skill actually
specifies (not a policy we designed)? If curated wikis were immune to
self-rewrite drift, that would undercut the case for the certificate/gating
machinery elsewhere in this repo; this experiment checks that they are not
immune, on real text.

## What it measures

- Compile wiki pages from a 75-claim tracked subset of SciFact-Open (`SUPPORT`
  claims) mixed into ~1500 filler PubMed abstracts.
- Stream new documents in, repeatedly **rewrite** each page ("update facts;
  newer supersedes" — the literal policy text of the Hermes llm-wiki
  SKILL.md), and re-check whether each of the 75 tracked claims is still
  judged `supported` by its page after each round.
- Two arms:
  - `plain` — rewrite only.
  - `harness` — rewrite + a check-and-rollback wrapper (revert a rewrite if
    it stops supporting the tracked claim).
- A **fresh-rebuild control**: at the end, throw away the rewrite history and
  recompile a wiki from scratch from the same current documents. The gap
  between `rebuild_retention` and the maintained wiki's final retention is
  the damage attributable to *iterated rewriting itself*, isolated from
  whatever the model would get right on a first pass.

## Files

- `wiki_decay.py` — the experiment driver: compiles pages, streams documents,
  rewrites pages each round, re-checks tracked claims. Arms: `plain`,
  `harness`. Includes the fresh-rebuild control.
- `build_subset.py` — builds `data/scifact_subset.json` from a local
  SciFact-Open checkout (already run; output is committed here so the
  experiment is self-contained — no separate download step needed).
- `plot.py` — renders the forgetting-curve figure and prints headline numbers
  from a `results/*.json` file.
- `post1_kaggle.ipynb` — a Kaggle notebook version (install → load code → run
  arms → plot), for running on a free T4 GPU instead of a local machine.
- `data/scifact_subset.json` — 75 tracked `SUPPORT` claims + ~1500 filler
  abstracts, real SciFact-Open data.
- `requirements.txt` — Python dependencies.
- `results/` — reference/expected output from a prior run (see "Status of the
  included results" below): `summary.json`, `harness_large_seed0.json`,
  `plain_large_seed0.json`, `plain_small_seed0.json`.

## Status of the included results — single seed, Run 2

The `results/` files in this directory are from a single run: **seed=0**,
retention judged by the **same model that did the rewriting**, at
**temperature 0** (no independent judge model, no multi-seed variance
estimate — see "Honesty notes" below).

They are **Run 2**, not the first attempt. Run 1 (not included here) had a
bug in how the fresh-rebuild control was computed — that bug inflated
`rebuild_retention` into a hollow ceiling that made rewriting look worse than
it actually was relative to a clean rebuild. That bug is fixed in the driver
copied here; the numbers in `results/summary.json` reflect the corrected
computation:

| Arm | r0 (start) | rT (after 12 rounds) | rebuild (fresh recompile at end) |
|---|---|---|---|
| `harness_large` | 32.0 | 29.3 | 53.3 |
| `plain_large` | 17.3 | 13.3 | 26.7 |
| `plain_small` | 22.7 | 2.7 | 49.3 |

(units: number of the 75 tracked claims still judged `supported`, i.e.
retention counts, not yet normalized to a fraction — see `plot.py` for the
normalized/percentage version.) In all three arms, `rebuild` retention is
well above the rewrite-degraded `rT`, i.e. iterated rewriting measurably
loses information beyond what a fresh compile from the same documents would
lose.

## How to run

### Run with Kilo Code (API — no GPU needed), recommended
Uses an OpenAI-compatible gateway with your own key, so it runs on any
machine (laptop, Kaggle CPU) with no GPU:
```bash
pip install requests scikit-learn matplotlib
export LLM_API_KEY=<your-kilo-key>                     # from app.kilo.ai -> API Keys
export LLM_BASE_URL=https://api.kilo.ai/api/gateway    # set explicitly, don't rely on a stale env var
M='nvidia/nemotron-3-super-120b-a12b:free'             # verified free + clean; $0-balance accounts can only use :free models
python wiki_decay.py --backend api --model "$M" --arm plain   --rounds 12 --seed 0
python wiki_decay.py --backend api --model "$M" --arm harness --rounds 12 --seed 0
python wiki_decay.py --backend api --model 'poolside/laguna-xs.2:free' --arm plain --rounds 12 --seed 0 --small
python plot.py
```
Notes carried over from the original development README:
- **Set `LLM_BASE_URL` explicitly** — a stale shell env var pointing
  elsewhere will hijack the call and produce a 401.
- **Free-tier only:** $0-balance Kilo accounts get 402 on paid models; use a
  `:free` model id. Free models tend to be reasoning models (~11-20s/call),
  so a 12-round run is roughly 2-3 hours; try `--rounds 8` for a faster first
  pass.
- Works with any OpenAI-compatible provider — just change `LLM_BASE_URL` +
  `--model` (Groq, OpenRouter, Google AI Studio, etc.).

Approximate call budget (batched judging is on by default): `plain` (12
rounds, 75 claims) ~420 calls, `harness` ~780 calls, `plain` small-model
~420 calls — roughly 1,200 calls total for the core chart (plain + harness).

### Run on Kaggle (local GPU, no API key)
1. New Notebook -> Settings: GPU T4 x2 (or P100), Internet ON.
2. Make this `external-validation/` folder available either by cloning the
   repo (set `REPO_URL` in cell 2 of `post1_kaggle.ipynb`) or uploading it as
   a Kaggle Dataset (cell 2 auto-finds it).
3. Run all cells. Minimum-viable = `plain` + `harness` (large model) +
   rebuild control -> the forgetting-curve chart.
4. Optional second session: the small-model arm.
5. Outputs land in `/kaggle/working/results` and `/kaggle/working/figures` —
   download from the Output tab.

Time/VRAM: a 4-bit ~8B model is roughly 5GB VRAM (a 14B model roughly 9GB).
One arm of 12 rounds x 75 claims takes roughly 1-2 hours on a T4.
Per-round checkpointing means a session cutoff (e.g. Kaggle's 12h limit)
won't lose progress — resume in a new session.

### Run locally (if you have a GPU)
```bash
pip install -r requirements.txt
python wiki_decay.py --arm plain   --model Qwen/Qwen3-8B   --rounds 12 --seed 0
python wiki_decay.py --arm harness --model Qwen/Qwen3-8B   --rounds 12 --seed 0
python wiki_decay.py --arm plain   --model Qwen/Qwen3-1.7B --rounds 12 --seed 0 --small
python plot.py
```

## What each number means

- **retention(t)** = fraction (or count, depending on output stage) of the 75
  tracked claims still judged `supported` by the wiki after `t` rewrite
  rounds. A falling line means forgetting.
- **rebuild_retention** = retention of a wiki recompiled from scratch at the
  end, from the same current documents, with no rewrite history. The gap
  between this and the maintained wiki's final retention is the damage
  attributable specifically to *rewriting*, not to the model's first-pass
  compilation quality.
- **plain vs harness** = without vs. with the check-and-rollback wrapper
  (revert a page edit if it breaks support for a tracked claim).

## Honesty notes (carried over from the original experiment writeup)

- Rewrite **prompts are faithful to** the Hermes llm-wiki SKILL.md update
  policy ("update facts; newer supersedes") and Karpathy's llm-wiki pattern.
  The skill itself is markdown instructions with no reference driver, so this
  experiment automates the *operator* (the loop that applies the policy each
  round) in a fixed, reproducible driver; it does not alter the policy text.
- Retention is judged by the **same model** that did the rewriting, at
  temperature 0 — this is a same-model self-judge, not an independent judge
  model. A judge-free string-match replication (e.g. against HoH-style
  ground truth) would be a natural follow-up to confirm the measured loss
  isn't a judge-consistency artifact; it has not been run for this
  experiment.
- **Single seed (seed=0).** No multi-seed variance estimate is included —
  the numbers in `results/` should be read as one run, not a distribution.
- Real (non-synthetic) data means the underlying model may recall a dropped
  fact from its own pretraining, which can *mask* measured loss — so the
  measured decay here is best read as a **conservative lower bound** on
  actual information loss from rewriting.
- Decay magnitude is model- and dataset-dependent; report the model and seed
  alongside any number quoted from this experiment.
