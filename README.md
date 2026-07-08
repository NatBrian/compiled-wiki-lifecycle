# The Compiled-Wiki Lifecycle

Research and reproduction code for **"The Compiled-Wiki Lifecycle: Certified Compilation, Maintenance, and Retraction for Evolving Corpora"**, a study of what an LLM-compiled knowledge store (a "wiki" an LLM writes and rewrites, in the pattern popularized by [Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) needs beyond a good initial compile: a way to know it's still current, a way to update it without silently destroying what it knows, and a way to remove something from it that must go.

**Paper**: [`paper/main.pdf`](paper/main.pdf) ([build instructions](#building-the-paper)), one paper, four stages, each independently validated.

## The four-stage lifecycle

| Stage | What it does | Headline result | Code |
|---|---|---|---|
| **Compile** | Resolve supersession at write time instead of retrieving raw chunks | SER $0.007$ vs. $0.177$–$0.280$ for RAG/LightRAG/RAPTOR baselines ($n{=}300$, HoH) | [`code/compile/`](code/compile/) |
| **Certify** | A finite-sample $(\delta,\varepsilon)$ statistical contract, noisy-judge-corrected, maintained incrementally | Maintained certificate matches full-audit validity at $40$–$47\%$ of the cost | [`code/certify/`](code/certify/) |
| **Maintain** | A trainable maintainer + a per-batch anytime-valid retention bound (LossGate) that rejects harmful updates | Trained maintainer $+0.127$ retention over best prompt ($p{=}0.009$); LossGate $+0.12$–$0.14$ over the same maintainer ungated | [`code/maintain/`](code/maintain/) |
| **Retract** | A membership-keyed write veto that survives an agent's own memory-consolidation loop | Correctness-gated deletion leaves $99.0\%$ of retracted facts recoverable; membership-gated leaves $0.0\%$, matching a never-ingested oracle | [`code/retract/`](code/retract/) |

Every number above is reported with its seed count, statistical test, and scope in the paper, see `paper/sections/07_limitations.tex` for what is *not* yet established (single-build variance in the compile stage, an untested half of a maintenance boundary law, and an open problem in certifying retraction of claims fused across multiple sources).

## Repo layout

```
paper/          LaTeX source, figures, bibliography, audit reports
code/
  compile/      HoH 7-arm comparison + label-free resolver + certificate scripts
                compile/external-validation/  independent robustness check on SciFact-Open,
                                               following a real third-party LLM-Wiki tool's update policy
  certify/      the (δ,ε) contract: gate, incremental maintenance, adaptivity-trap detection
  maintain/     trained-maintainer/  SFT-trained maintenance + reward-gaming boundary law
                lossgate/            per-batch anytime-valid retention bound
  retract/      agent-memory/           backflow phenomenon + membership-veto gate (primary result)
                leakage-refusal-gate/   serving-time leakage refusal (secondary result, see its
                                        README for an important framing caveat before citing it)
  shared/       utilities reused across ≥2 stages (stats, LLM client, data loading)
docs/
  REPRODUCE.md    per-stage run instructions
  LIMITATIONS.md  consolidated, stage-by-stage
results/          small final result files backing the paper's tables
```

## Quickstart

Each stage needs a locally-served model (vLLM, OpenAI-compatible) and downloads its own datasets on first run, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for exact commands, ports, and expected runtimes per stage. There is no single "run everything" script by design: the four stages use different models, GPU budgets, and (for `retract/agent-memory/letta_ladder.py`) optional external services (Letta, Postgres+pgvector) that most reproductions won't need.

```bash
git clone https://github.com/NatBrian/compiled-wiki-lifecycle.git
cd compiled-wiki-lifecycle
pip install -r requirements.txt   # consolidated; see docs/REPRODUCE.md for which subset a given stage needs
```

## Building the paper

```bash
cd paper
latexmk -pdf main.tex   # requires a TeX Live distribution (TinyTeX is sufficient)
```

## Honesty notes

- None of this work has been externally peer-reviewed. Every headline number in the paper was checked against the underlying experiment logs and analysis scripts; `paper/PAPER_CLAIM_AUDIT.md` and `paper/CITATION_AUDIT.md` document the verification done before publication.
- `code/retract/leakage-refusal-gate/README.md` carries a specific, load-bearing caveat: only its `e2_leakage.py` result is used in the paper. The other scripts in that directory implement a formal-criterion approach found unsound for the case it targets, they're included for research transparency, not as validated results.
- LICENSE: MIT.
