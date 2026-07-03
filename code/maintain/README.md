# maintain/ — two mitigations for wiki rewrite-collapse

Both experiments in this directory start from the same disease: an LLM-compiled wiki page that
gets iteratively rewritten to absorb new documents (constant-corpus maintenance) loses facts
over time — "rewrite collapse" — because each rewrite is an independent lossy compression
step, and losses compound. There are two structurally different responses to that:

- **`trained-maintainer/`** changes *what does the rewriting*: distill a maintainer policy
  (SFT on rejection-sampled rewrites) that is better at the rewrite step itself than any
  prompt we could hand-write.
- **`lossgate/`** changes *nothing about the maintainer* and instead wraps *any* maintainer
  (prompt or trained policy) in a gate that re-checks each rewrite before committing it, and
  emits a certified per-batch bound on how much retention could have been lost.

They are complementary, not competing: LossGate is explicitly policy-agnostic and its E4
experiment gates around a trained LoRA adapter (`trained-maintainer`'s output family) to show
composition. Read `trained-maintainer/README` intuition below first, then `lossgate/`.

## The shared decay decomposition: budget crowding vs. rewrite damage

A maintained wiki page can lose facts for two mechanically different reasons, and every
result in both experiments is designed to isolate one from the other:

1. **Budget crowding** — the page has a word cap. As more distinct facts accumulate on the
   same page, later facts compete for a shrinking budget and something has to be dropped or
   compressed just because the page is *full*. This is a capacity problem, not a rewriting
   problem — it would happen even with a perfect, lossless editor.
2. **Rewrite damage** — every incorporation of a new document is a full LLM rewrite of the
   page text, and each rewrite is an independent lossy compression pass. A fact that survived
   batch $t$ can be silently dropped, paraphrased into ambiguity, or merged away at batch
   $t{+}1$, purely from repeated lossy rewriting, independent of how much room is left.

Both experiments hold the corpus size **constant** (`--replace_fillers 1`, the default): each
maintenance batch swaps a fixed number of filler documents out and the same number in, so the
page's face never permanently grows just because time passes. This isolates the *rewrite
damage* clock (cumulative page rewrites) from the *budget crowding* clock (page length) —
`hazard_analysis.py` (trained-maintainer) explicitly fits a discrete-time hazard model with
`page_rewrites` and `page_words` as separate covariates so the two effects don't get
conflated in the reported half-life numbers.

## `trained-maintainer/` — SFT-distilled faithful maintainer

**Result (honest framing).** On the matched, seeded 12-batch constant-corpus tournament
(SciFact-Open, Qwen2.5-14B, 5 seeds), a maintainer policy trained by rejection-sampling SFT
(best-of-$k{=}8$ candidates scored on doc-claim retention, LoRA r=16) reaches corrected
retention

$$R_{12} = 0.523 \pm 0.037 \quad\text{vs. the strongest hand-written prompt (\texttt{anchored}) } 0.397 \pm 0.059,$$

a gain of $+0.127$ (bootstrap 95% CI $[0.067, 0.187]$, Welch $p=0.009$), recovering **90%**
of the maintained-vs-clean-rebuild retention gap (vs. 43% for the best prompt). The trained
policy runs at **single-sample deployment cost** — no extra inference-time sampling — because
the best-of-$k$ search only happens once, during SFT data generation.

**On the "Pareto-optimal" claim — corrected.** An earlier pass of the internal draft claimed
the trained maintainer is *Pareto-optimal* on the preservation–incorporation frontier
(higher retention **and** at least as much incorporation as every prompt arm). That is an
overclaim: the incorporation-rate margin between trained (0.088) and the `anchored` prompt
(0.083) is **statistically null** (Welch $p=0.54$; trained-vs-`proto` incorporation is
$p=0.15$). The data support the narrower, defensible claim actually used in this README and
that should be used anywhere these numbers are cited: **the trained maintainer dominates on
retention at incorporation no lower than the prompt** — not a genuine two-axis Pareto win,
because the incorporation side of that comparison doesn't clear significance. Do not repeat
"Pareto-optimal" without this caveat.

**Currency safety (does training create entrenchment?)** `currency_safety.py` reruns the
staggered-supersession protocol (HoH pairs, explicit-REPLACE rewrites routed through the
trained maintainer) to check whether a policy trained to *preserve* facts also learns to
*resist legitimate updates*. This is reported as a separate, explicit experiment rather than
assumed away — a maintainer that hoards is not "more faithful," it's differently broken.

**Goodhart-in-training check.** `goodhart_eval.py` replays the exact training-style chains
under the trained policy and measures retention separately on `P_train` (claims that were
literally rewarded during SFT), `P_held` (other claims on the same page, never rewarded), and
`P_fresh` (a claim never seen anywhere near training). A big `P_train - P_held` gap after
training (relative to the vanilla maintainer replaying the same chains) would mean the
reward signal got gamed rather than generalizing; `hazard_analysis.py` and `currency_safety.py`
give two independent, zero-additional-GPU angles on the same mechanism question from existing
run logs.

**Files** (`code/maintain/trained-maintainer/`):

| file | role |
|---|---|
| `p4_protocol.py` | main maintenance-arm harness — constant-corpus rewrite loop, arm = (prompt, served model); emits the `results_*.json` retention timelines |
| `sft_datagen.py` | rejection-sampling SFT data generation (best-of-$k$ candidate rewrites scored on doc-claim retention) |
| `sft_train.py` | LoRA SFT trainer (TRL `SFTTrainer`) over the generated data |
| `goodhart_eval.py` | Goodhart-in-training replay eval (P_train / P_held / P_fresh) |
| `hoh_mc.py` | judge-free replication on the external HoH benchmark (string-match retention, no LLM judge) |
| `currency_safety.py` | staggered-supersession / entrenchment check for the trained maintainer |
| `hazard_analysis.py` | discrete-time hazard model (page_rewrites vs. page_words covariates) — the budget-crowding-vs-rewrite-damage decomposition, zero GPU (reads existing run logs) |
| `p4_analysis.py` | decay-law fitting, Kaplan–Meier survival, tournament table across all arms |
| `summarize_v2.py` | single source of truth for the headline numbers above (reads `results/*.json`, prints + dumps `results/summary_v2.json`) |
| `make_figs.py` | figures from `results/analysis_*.json` |
| `make_jobs.py` / `run_pool.py` | generates and runs the batch job manifest (crash-safe throttled pool) for the full experiment sweep |
| `seeded_client.py` | vLLM client subclass that injects a per-run `seed` into every chat request (variance-honesty fix — the shared `oai_client.VLLM` does not pass `seed`, so distinct `--seed` values wouldn't otherwise be independent sampling replications) |
| `serve.sh`, `serve_eval.sh` | launch the local vLLM server(s) (base model + LoRA adapters) this code talks to |
| `detached_run.sh` | crash-safe launcher (`setsid`, retries until the result file exists) |

**CLI usage** (from `p4_protocol.py`'s actual `argparse`):

```bash
cd code/maintain/trained-maintainer

# 1. serve the base model (+ any LoRA adapters you've trained) on your own GPUs
./serve.sh                                   # base only
./serve.sh b1-trained=results/lora_b1_preserve   # base + one adapter, name=path

# 2. run one maintenance-arm timeline (vanilla / conservative / anchored, or a served adapter)
python p4_protocol.py --arm anchored --seed 0 \
  --data /path/to/scifact-open/data --out results/results_a1_anchored_seed0.json
python p4_protocol.py --arm vanilla --maintainer_model b1-trained --seed 0 \
  --out results/results_b1_trained_seed0.json

# same launcher pattern, detached so a client crash can't kill the run:
./detached_run.sh a1_anch_s0 results/results_a1_anchored_seed0.json -- \
  python p4_protocol.py --arm anchored --seed 0 --out results/results_a1_anchored_seed0.json

# 3. generate SFT data (rejection sampling) and train
python sft_datagen.py --mode preserve --tag b1_preserve --data /path/to/scifact-open/data
python sft_train.py --data results/sft_b1_preserve.jsonl --out results/lora_b1_preserve

# 4. analyze + summarize + plot
python p4_analysis.py --pattern "results_a1_*_seed*.json" --tag main
python summarize_v2.py
python make_figs.py
```

`--data` defaults to `<repo_root>/benchmarks/scifact-open/data`, an **external dataset not
bundled in this repo** — fetch SciFact-Open separately and point `--data` at it (or override
per-invocation as shown above). `--cal_file` (judge-noise calibration) defaults to
`code/certify/contract/results_e1.json`, which **is** bundled (see `code/certify/`).

## `lossgate/` — policy-agnostic certified retention bound

**The idea.** LossGate wraps *any* maintainer (prompt or trained policy) with a gate: before
committing a page rewrite, re-extract the facts that page is supposed to support from a
held-out **GATE** probe pool, and **roll back** the rewrite if it destroys more than $\tau$
previously-confirmed facts. Each batch this produces a certified per-transition bound

$$R(t) \ge R(t-1) - b_t \quad \text{at confidence } 1-\alpha,$$

where $b_t$ is a Clopper–Pearson upper bound on the destroyed fraction, computed on the GATE
pool, and then *validated* against a disjoint **VALID** pool the gate never touches — so
certificate coverage (E1) is a genuine test, not self-referential. `p5_stream.py` composes
the per-batch bounds into a single **anytime-valid** (time-uniform) stream-level LCB via a
Basel-weighted confidence-sequence construction, so the guarantee holds simultaneously over
an unbounded maintenance stream, not just at one fixed horizon.

**Make-or-break result (A1).** Gating strictly improves retention over the ungated policy it
wraps, at comparable currency (incorporation rate):

$$\text{LossGate(vanilla)} - \text{vanilla}: \; +0.12 \text{ retention } (3.4\sigma,\text{ positive in }6/7\text{ seeds})$$
$$\text{LossGate(vanilla)} - \text{conservative}: \; +0.14 \text{ retention } (2.7\sigma,\text{ positive in }5/7\text{ seeds})$$

Critically, this is not "win by freezing rewrites" (the pre-registered kill condition): the
gated arm keeps comparable incorporation to the ungated policy it wraps.

**On certificate tightness — read this before citing the bound.** *Validity* (the bound
actually holds at the stated confidence) and *tightness* (how far the bound is from the
realized retention) are different properties, and only one of them is guaranteed
unconditionally. On the small SciFact-Open GATE pool ($n \approx 15$ confirmed facts/batch)
the cumulative bound is **loose** — it is still a valid lower bound, just not a tight one. On
the larger judge-free HoH pool ($n{=}120$) the composed stream LCB stays strictly positive
and reasonably tight across all 12 batches (final LCB 0.07–0.13 vs. realized retention 0.86).
**The guarantee is valid in both regimes; only its tightness scales with the audit budget.**
Do not read "the bound is loose on SciFact" as "the bound is wrong on SciFact" — it is a
correct, conservative statement that happens not to be very informative at small $n$.

**Scope, disclosed not hidden.** All main results use a single $14$B maintainer scale — a
second model scale was **not run**, and this was a deliberate choice, not a dropped result we
are hiding: running a second scale would have meant contending for shared GPUs with other
users' jobs, which this project's operating rule (never disrupt others' running GPU jobs)
rules out. The paper's own Limitations section says this explicitly. Similarly, **E4**
(gating around a trained LoRA adapter, showing policy-agnostic composition with
`trained-maintainer/`'s output) and **E5** (the $\tau$ over-conservatism sweep) are each
**single-seed** — they demonstrate the mechanism (composition works; relaxing $\tau$
monotonically trades rollback aggressiveness for currency, mean rollbacks/run
$5\to0\to0$ for $\tau=0,1,2$) rather than providing a seed-averaged effect-size estimate. Treat
E4/E5 as existence proofs, not as precision claims on par with the 7-seed A1 make-or-break
result.

**Files** (`code/maintain/lossgate/`):

| file | role |
|---|---|
| `p5_lossgate.py` | main gate-and-rollback harness on SciFact-Open — arms `vanilla`/`conservative`/`anchored` (no gate) and `lossgate_vanilla`/`lossgate_conservative`/`lossgate_anchored` (gated); emits the per-batch certificate `b_t` alongside retention |
| `p5_hoh.py` | judge-free replication (E6) on the external HoH benchmark — string-match retention/gate, so `b_t` carries no judge-noise correction (exact binomial CP bound) |
| `p5_stream.py` | anytime-valid composition of per-batch bounds into a stream-level LCB (union-bound and confidence-sequence variants) |
| `p5_analysis.py` | A1 make-or-break verdict + E1 certificate coverage + E2 stream-vs-static composition, from `results/a1_*.json` |
| `p5_analysis2.py` | comprehensive refinement analysis (seed CIs, E2-rescue on the large HoH pool, E4 LoRA gate, E5 $\tau$ sweep) $\to$ `results/analysis2.json` |
| `fill_numbers.py` | populates the paper's LaTeX result macros from `results/analysis.json` |
| `make_figs.py` | A1 frontier / E2 stream-vs-static figures |
| `run_a1.sh` | launches the full A1 make-or-break sweep (4 arms $\times$ up to 7 seeds, two parallel server queues, detached) |
| `run_e6.sh` | launches the E6 judge-free HoH replication sweep |
| `run_e45.sh` | launches E4 (gate around LoRA maintainer) + E5 ($\tau$ sweep) |
| `run_queue.sh` | single sequential (arm,seed) queue on one port — the primitive `run_a1.sh` builds on |
| `detached_run.sh` | crash-safe launcher (`setsid`, retries until the result file exists) |
| `example_results/` | 3 representative result JSONs checked in as a concrete output-shape example (all well under 500KB): `analysis.json` (main A1/E1/E2 summary), `analysis2.json` (refinement analysis with seed CIs), `a1_lossgate_vanilla_seed0.json` (one raw per-seed run) |

**CLI usage** (from the actual `argparse` in `p5_lossgate.py` and the shell wrappers):

```bash
cd code/maintain/lossgate

# one gated run (arm = lossgate_<base-prompt>; tau=0 is the strict/reported operating point)
python p5_lossgate.py --arm lossgate_vanilla --tau 0 --seed 0 --port 8102 \
  --n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --track_incorp 12 \
  --out results/a1_lossgate_vanilla_seed0.json

# ungated baseline for the same comparison
python p5_lossgate.py --arm vanilla --seed 0 --port 8102 \
  --n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --track_incorp 12 \
  --out results/a1_vanilla_seed0.json

# full A1 sweep (matches the paper's headline numbers), judge-free E6 replication, E4+E5
./run_a1.sh
./run_e6.sh
./run_e45.sh      # needs an adapter served as 'maintainer-lora' on the target port

# analyze
python p5_analysis.py     # -> results/analysis.json  (A1 verdict, E1 coverage, E2)
python p5_analysis2.py    # -> results/analysis2.json (seed CIs, E4, E5)
python fill_numbers.py    # populate paper/latex/main.tex result macros
python make_figs.py
```

`--data` (SciFact-Open corpus) and `--cal_file` (judge-noise calibration) resolve the same way
as in `trained-maintainer/` above: `--data` is an external dataset you must fetch yourself;
`--cal_file` defaults to the bundled `code/certify/contract/results_e1.json`.

## Shared imports

`trained-maintainer/` and `lossgate/` both depend on a common compile/judge/stats library
that lives at [`code/shared/`](../shared/) (re-exported from
[`code/certify/contract/`](../certify/contract/), which also carries the data assets —
`e5_data.json`, `results_e1.json` — that a few scripts here reference by path). Every script
in both directories does:

```python
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))      # .../compiled-wiki-lifecycle/code
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage's data assets (e5_data.json, results_e1.json)
```

`trained-maintainer/` and `lossgate/` do **not** import from each other's code directly (the
two only reference each other in comments/docstrings — LossGate's E4 experiment
composes with a trained adapter by *serving* it under vLLM and passing its name via
`--maintainer_model`, not by importing `trained-maintainer/`'s Python). All shell scripts
use `cd "$(dirname "$0")"` so they run correctly regardless of where the repo is cloned.

## Not bundled

- **SciFact-Open** corpus data (`--data` default `<repo_root>/benchmarks/scifact-open/data`)
  — fetch it yourself; not included in this repo.
- **LoRA checkpoints.** `trained-maintainer/`'s trained adapters are not included (multi-GB
  binary artifacts); `sft_train.py` reproduces them from `sft_datagen.py`'s output. `results/`
  directories with raw per-run JSONs and any trained checkpoints are likewise not bundled,
  except for the 3 small representative files under `lossgate/example_results/`.
- A **local vLLM server** exposing an OpenAI-compatible endpoint for Qwen2.5-14B is required
  to run anything here end-to-end; nothing in this directory can be exercised without it.
