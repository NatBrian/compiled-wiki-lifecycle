# Compile stage

This directory holds the driver code for the **compile** stage of the pipeline:
turning a raw, growing corpus of evidence documents into a queryable knowledge
source, and comparing that against six baseline "arms" on questions whose
correct answer changes over time (facts get superseded — a later document
overwrites an earlier one on the same underlying concept).

## The six arms

All arms read from the *same* pooled evidence corpus (multiple HoH items'
documents interleaved into one shared pool, so systems must both retrieve the
right concept **and** disambiguate current-vs-stale within it, not just recall
a small isolated context):

| Arm | What it does |
|---|---|
| `closed_book` | Parametric-only baseline — no retrieval context at all. |
| `full_dump_cic` | LOFT-style Corpus-in-Context: every doc, ID-tagged, dumped into a long-context reader (Qwen3-Coder-30B, 256K ctx). No curation. |
| `vector_rag` | Standard dense (bge) retrieval, top-k over the pooled corpus. Stale and current versions of a fact coexist and compete at retrieval time. |
| `lightrag` | Third-party graph-RAG system ([LightRAG](https://github.com/HKUDS/LightRAG)), run faithfully against its own ingestion/query API. |
| `raptor` | Third-party hierarchical-summarization retriever ([RAPTOR](https://github.com/parthsarthi03/raptor)), run faithfully against its own API. |
| `wiki_karpathy` | The compiled-wiki arm: one page per concept, overwritten in ingestion order so only the current fact is ever on the page (Karpathy "LLM wiki" pattern). FTS5 keyword navigation, no embeddings. |
| `resolve_free` | The label-free supersession **resolver** (`src/resolver.py`) bolted onto the `vector_rag` retriever output: clusters retrieved chunks by *text similarity* (not a gold concept id) and keeps the latest-*ingested* member of each cluster (not a gold version tag). This is the deployable lever — it needs no oracle metadata, so it can sit on top of any retriever. |

There is also a `mem0` / `graphiti` agent-memory arm (`run_mem0.py`,
`run_graphiti.py`) and a per-item (non-pooled) HoH variant of each arm
(`hoh_*.py`) used for smaller-scale / isolation-context runs before the pooled
setup above became the headline comparison.

## Directory layout

```
code/compile/
  real/            hand-written driver scripts (one per arm / experiment)
    pooled_hoh.py       main pooled-corpus driver: closed_book, full_dump_cic,
                         vector_rag, wiki_karpathy, resolve_free
    pooled_lightrag.py  pooled-corpus driver for the LightRAG arm
    pooled_raptor.py    pooled-corpus driver for the RAPTOR arm
    hoh_lightrag.py, hoh_raptor.py, hoh_mem0.py, hoh_longcontext.py
                         per-item (non-pooled) variants of the same arms
    run_lightrag.py, run_raptor.py, run_mem0.py, run_graphiti.py
                         earlier per-arm single-system runners
    real_core.py         shared core: HoH dataset loading, chat-completion
                         helper, corpus streaming, result-record schema
    loft_run.py          LOFT (long-context) corpus-in-context reader driver
    assemble.py, assemble_scale.py
                         corpus assembly / scale-sweep helpers
    longcontext.py        long-context reader helper used by loft_run.py
    pool_run.sh, hoh_chain.sh, hoh_one.sh, loft_chain.sh,
    run_real_chain.sh, sweep_run.sh
                         orchestration wrappers: retry-until-checkpointed
                         loops over one or more arms
    detached_run.sh       launches any command via `setsid` so a crashed
                         host shell/agent process can't kill a long run;
                         retries until its expected result file exists
  src/
    resolver.py          the label-free supersession resolver (P2): clusters
                         retrieved chunks by embedding similarity and keeps
                         the latest-ingested member per cluster — no gold
                         concept id, no gold version tag
  scripts/               certificate / analysis / figure scripts (post-hoc,
                         run over the results/*.json produced by real/)
    certify.py            builds the retention/staleness-elimination
                         certificate comparing arms
    strong_stale.py        stricter staleness-detection scoring pass
    judge_noise_final.py   LLM-judge noise/agreement estimate
    make_tables.py         renders LaTeX result tables from result JSONs
    fig_certificate.py     renders the certificate figure
    audit_numbers.py       cross-checks headline numbers back to raw results
  external-validation/   independent replication on a different dataset with
                         a different (real, published) tool — see its own README
```

Everything under `real/` and `src/` is hand-written for this project. It is
**not** a vendored copy — the actual third-party libraries it calls
(`lightrag-hku`, `raptor`, `mem0`, `graphiti-core`, `sentence-transformers`,
`datasets`, `openai`) are ordinary pip dependencies, installed into
per-framework virtualenvs during development (not bundled in this repo; see
"Setup" below).

## Setup

1. **A local vLLM (or any OpenAI-compatible) server** serving the reader
   model, by default `Qwen2.5-14B-Instruct`:
   ```bash
   export AGENT_MODEL=qwen2.5-14b
   export AGENT_URL=http://127.0.0.1:8101/v1
   ```
   The `full_dump_cic` / LOFT long-context arm additionally expects a
   long-context model (default `qwen3-coder-30b`) on a second port:
   ```bash
   export DUMP_MODEL=qwen3-coder-30b
   export DUMP_URL=http://127.0.0.1:8102/v1
   ```
   Any server that speaks the OpenAI `chat/completions` API on those URLs
   works — vLLM, TGI, llama.cpp server, etc. All scripts read these from the
   environment with the above as defaults.

2. **Dataset** — auto-downloads on first run via HuggingFace `datasets`:
   ```python
   datasets.load_dataset("russwest404/HoH-QAs", split="train")
   ```
   No manual download step; just make sure the machine has network access
   (or a populated HF cache) the first time a driver runs.

3. **Per-framework environments.** The original development setup used one
   virtualenv per third-party framework (`real/venvs/lightrag`,
   `real/venvs/raptor`, ...) because several of these libraries pin
   conflicting dependency versions. Those venvs are **not** included in this
   repo (they're ~850MB of vendored packages). Recreate what you need, e.g.:
   ```bash
   pip install lightrag-hku sentence-transformers openai datasets  # for the lightrag arm
   pip install raptor-py sentence-transformers openai datasets     # for the raptor arm (TODO: confirm exact PyPI name — the
                                                                    #   original venv installed RAPTOR from a source checkout,
                                                                    #   not necessarily this package)
   pip install mem0ai sentence-transformers                        # for the mem0 arm (TODO: confirm exact PyPI name)
   pip install graphiti-core kuzu sentence-transformers            # for the graphiti arm
   ```
   The `wiki_karpathy`, `vector_rag`, `resolve_free`, `closed_book`, and
   `full_dump_cic` arms need only common packages (`openai`,
   `sentence-transformers`, `datasets`, stdlib `sqlite3` for FTS5) and run
   fine from a single environment.

## Running

The headline pooled-corpus comparison (`pooled_hoh.py`):
```bash
cd code/compile
python real/pooled_hoh.py closed_book 300      # single arm, N=300 pooled items
python real/pooled_hoh.py all 300               # all arms in one process
```
The third-party-framework arms live in their own drivers, invoked from their
respective venvs:
```bash
real/venvs/lightrag/bin/python real/pooled_lightrag.py 300
real/venvs/raptor/bin/python real/pooled_raptor.py 300
```
`real/pool_run.sh` wraps a set of arms in a retry-until-checkpointed loop
(each arm checkpoints per item to `results/_ckpt_pool_<arm>.jsonl` and
resumes on restart — the underlying host can SIGKILL a long-running process
without losing progress):
```bash
real/pool_run.sh 300 closed_book vector_rag wiki_karpathy resolve_free
```
For anything you want to survive a crash of the *launching* shell/agent
itself (not just the benchmark process), wrap it in `detached_run.sh`, which
launches via `setsid` (new session, no controlling terminal, reparented to
init) and retries until the expected result file appears:
```bash
real/detached_run.sh pool_raptor results/results_pool_raptor.json -- \
  real/venvs/raptor/bin/python real/pooled_raptor.py 300
```

After results land under `results/`, the analysis/certificate scripts in
`scripts/` turn them into the paper's tables and figures:
```bash
python scripts/certify.py
python scripts/strong_stale.py
python scripts/judge_noise_final.py
python scripts/make_tables.py
python scripts/fig_certificate.py
python scripts/audit_numbers.py
```

## Environment variables (summary)

| Var | Default | Meaning |
|---|---|---|
| `AGENT_MODEL` / `AGENT_URL` | `qwen2.5-14b` / `http://127.0.0.1:8101/v1` | reader model used by most arms |
| `DUMP_MODEL` / `DUMP_URL` | `qwen3-coder-30b` / `http://127.0.0.1:8102/v1` | long-context reader for `full_dump_cic` |
| `POOL_K` | `4` | retrieval top-k for `vector_rag` / `resolve_free` |
| `RESOLVE_THRESH` | `0.86` | similarity threshold for the resolver's clustering (`src/resolver.py`) |
| `RESOLVE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model used by the resolver |
| `BENCH_STAMP` | `0` | `1` = version-stamped regime (recency label visible); default is version-anonymous (headline regime) |
| `BENCH_STATIC` | `0` | `1` = static corpus (current-only docs, no superseded versions) |
| `TAG_SUFFIX` | `""` | suffix appended to result tags, used for reader-size sweeps |

Not every script uses every variable above — see each file's header comment
for the specific set it reads.
