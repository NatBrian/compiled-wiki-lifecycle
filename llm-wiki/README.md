# LLM Wiki

**A minimal, self-compiling knowledge wiki powered by LLMs.**

LLM Wiki is a CLI tool that reads research notes in `sources/`, uses an LLM to extract key concepts, generates interconnected wiki pages, and lets you query the resulting knowledge base, all without a database, vector store, or internet (when using a local model).

Think of it as an automatic Wikipedia for your personal research. You provide the source material; the LLM does the writing, organizing, and cross-referencing. You just read and ask questions.

> **Core philosophy** (inspired by [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)):
> *"The LLM writes and maintains the wiki; the human reads and asks questions.
> The wiki is a persistent, compounding artifact."*

---

## Table of Contents

1. [What is an LLM Wiki?](#1-what-is-an-llm-wiki)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Layout](#3-directory-layout)
4. [The Compile Pipeline (Step by Step)](#4-the-compile-pipeline-step-by-step)
5. [The Query Pipeline](#5-the-query-pipeline)
6. [Installation](#6-installation)
7. [Configuration](#7-configuration)
8. [Usage Guide](#8-usage-guide)
9. [Module Reference](#9-module-reference)
10. [Edge Cases & Important Details](#10-edge-cases--important-details)
11. [What's Not Included](#11-whats-not-included)
12. [Comparison with Other LLM Wikis](#12-comparison-with-other-llm-wikis)

---

## 1. What is an LLM Wiki?

### The Problem

You have research notes, paper summaries, meeting transcripts, dozens of markdown files scattered in a folder. When you need to recall something, you grep frantically, re-read entire documents, or just give up. Your knowledge doesn't compound; it decays.

### The Solution

LLM Wiki takes your source documents and:

1. **Extracts concepts**, reads each source and identifies 3–8 key ideas
2. **Generates wiki pages**, writes an encyclopedic page for each concept
3. **Links everything together**, connects related pages with `[[wikilinks]]`
4. **Answers questions**, selects relevant pages and synthesizes answers

The output is a **persistent, navigable wiki** that grows smarter every time you add sources. The LLM does the writing and maintenance; you just add source material and ask questions.

### A Concrete Example

```
Before (10 messy SciFact abstracts in sources/):
  bayesian-measures-of-model-complexity.md
  efficient-targeting-of-expressed-and-silent-genes.md
  ...

After one compile (49 interconnected wiki pages):
  wiki/concepts/
    bayesian-model-complexity.md
    deviance-information-criterion.md
    zfn-mediated-genome-editing.md
    likelihood-ratios-in-diagnostic-accuracy.md
    ...
  wiki/index.md          ← auto-generated table of contents
  .llmwiki/state.json    ← tracks what changed since last compile
  log.md                 ← activity journal
```

And from then on, you ask questions in plain English:

```bash
llm-wiki query "What is the Deviance Information Criterion?"
→ "The DIC is a Bayesian model comparison metric used for model selection..."
```

---

## 2. Architecture Overview

### Mental Model

Think of the system as three layers:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: YOU                                          │
│  Drop .md files into sources/ and ask questions         │
├─────────────────────────────────────────────────────────┤
│  Layer 2: The Wiki (wiki/concepts/*.md)                 │
│  LLM-generated pages, cross-linked with [[wikilinks]]   │
├─────────────────────────────────────────────────────────┤
│  Layer 1: The Compiler                                  │
│  Detects changes → extracts concepts → generates pages  │
│  → resolves links → builds index → logs everything      │
└─────────────────────────────────────────────────────────┘
```

### High-Level Pipeline

```
  sources/*.md
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Change   │───▶│ 2. Concept   │───▶│ 3. Merge by  │───▶│ 4. Generate  │
│  Detection  │    │  Extraction  │    │    Slug      │    │    Pages     │
│ (SHA-256)   │    │ (LLM tool)   │    │ (deduplicate)│    │ (LLM writes) │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
      ▲                                                           │
      │                                                           ▼
      │                                             ┌──────────────┐
      │                                             │ 5. Resolve    │
      │                                             │    Wikilinks  │
      │                                             │ (deterministic│
      │                                             │  regex)      │
      │                                             └──────┬───────┘
      │                                                    │
      │                                                    ▼
      │                                             ┌──────────────┐
      │                                             │ 6. Generate   │
      │                                             │    Index     │
      │                                             │ (wiki/index) │
      │                                             └──────┬───────┘
      │                                                    │
      │                                                    ▼
      │                                             ┌──────────────┐
      │                                             │ 7. Log &     │
      │                                             │    Save      │
      │                                             │    State     │
      │                                             └──────────────┘
      │
      └────────────────────── Query Pipeline ──────────────────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │ 1. Select Pages   │
                           │ (3 tiers, with   │
                           │  wikilink graph  │
                           │  expansion)      │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ 2. Answer         │
                           │ (LLM synthesizes) │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ 3. Save (optional)│
                           │ wiki/queries/     │
                           └──────────────────┘
```

### Two Providers

The tool supports two LLM providers, selected via the `LLMWIKI_PROVIDER` environment variable:

```
LLMWIKI_PROVIDER=anthropic ──── Anthropic API (Claude Sonnet 4, default)
LLMWIKI_PROVIDER=openai   ──── OpenAI-compatible API (vLLM, local models, etc.)
```

The **Anthropic provider** sends requests to `api.anthropic.com`. The **OpenAI provider** sends requests to a configurable base URL (default `http://localhost:8000/v1`), this is what you use with local models served by vLLM, Ollama, or any OpenAI-compatible endpoint.

Both providers support:
- **Tool calls** for structured concept extraction and page selection
- **Freeform completion** for page generation and answer synthesis

### File Locking (Crash Safety)

The compiler uses a **file-based lock** (`.llmwiki/lock`) to prevent concurrent compiles from corrupting the state. If the lock cannot be acquired within 30 seconds, the compile aborts with `"Could not acquire lock"`. The lock is always released in a `finally` block, so even crashes during compile will clean up on the next run.

---

## 3. Directory Layout

```
my-wiki/
├── sources/             # ← YOU put your markdown files here
│   ├── paper-1.md
│   └── notes.md
├── wiki/
│   ├── concepts/        # ← LLM-generated wiki pages (1 file = 1 concept)
│   │   ├── self-attention.md
│   │   └── transformer.md
│   ├── queries/         # ← Saved query answers (optional, from --save)
│   │   └── query-what-is-mamba-20260708_120000.md
│   └── index.md         # ← Auto-generated table of contents
├── .llmwiki/
│   ├── state.json       # ← SHA-256 hashes + concept mappings per source
│   └── lock             # ← Prevents concurrent compiles
└── log.md               # ← Append-only activity journal
```

### What Goes in `sources/`

- **Only `.md` files**, other extensions are ignored
- Each file should be at least **50 characters** (`MIN_SOURCE_CHARS`)
- Recommended max: **100,000 characters** (`MAX_SOURCE_CHARS`)
- Filename examples: `2024-07-paper.md`, `meeting-notes.md`, `attention-is-all-you-need-summary.md`
- Put whatever you want, paper summaries, notes, transcripts, analysis

### What Comes Out in `wiki/concepts/`

Each generated page has YAML frontmatter and markdown body:

```markdown
---
title: Self-Attention
summary: A mechanism for weighting the importance of different elements in an input sequence
sources:
  - attention-paper.md
  - my-notes.md
kind: concept
createdAt: 2026-07-08T12:00:00Z
updatedAt: 2026-07-08T12:00:00Z
confidence: 0.95
provenanceState: extracted
tags:
  - deep-learning
  - nlp
---

Self-Attention is a mechanism that computes a weighted sum of values based on
query-key similarity. It is the core building block of the
[[transformer|Transformer]] architecture ^[attention-paper.md:1-3].

## Sources
- attention-paper.md
```

**Frontmatter fields:**

| Field | Description |
|---|---|
| `title` | Human-readable concept name |
| `summary` | One-line description |
| `sources` | Which source files contributed to this page |
| `kind` | Always `"concept"` |
| `createdAt` | ISO timestamp of first creation (preserved across rewrites) |
| `updatedAt` | ISO timestamp of last modification |
| `confidence` | 0.0–1.0 score from the LLM |
| `provenanceState` | `"extracted"` (single source) or `"merged"` (multiple sources) |
| `tags` | Optional labels for grouping |
| `orphaned` | Set to `true` when no other page links to this one |

### What's in `.llmwiki/state.json`

```jsonc
{
  "version": 1,
  "sources": {
    "paper-1.md": {
      "hash": "abc123def...",           // SHA-256 of the file content
      "concepts": ["attention", "transformer"],  // slugs extracted from this source
      "compiledAt": "2026-07-08T14:00:00Z"
    }
  },
  "frozenSlugs": ["shared-concept"],     // slugs preserved when source deleted
  "indexHash": ""
}
```

This file is the **brain** of the incremental compile, it remembers what was extracted from each source so the next compile only processes what changed.

---

## 4. The Compile Pipeline (Step by Step)

### Step 1: Change Detection

The compiler scans `sources/` and compares each file's SHA-256 hash against `state.json`:

| Situation | What happens |
|-----------|-------------|
| New source: no entry in state | Full extraction |
| Changed source: hash differs | Full extraction |
| Unchanged source: hash matches | Skipped (no LLM calls) |
| Deleted source: in state, not on disk | Orphan pages or freeze shared concepts |

This is what makes the second compile **fast**. Unchanged sources make zero LLM calls.

### Step 2: Concept Extraction

For each new/changed source file, the LLM is asked:

> *"Extract 3–8 key concepts from this document. Use the `extract_concepts` tool."*

The LLM returns structured data via a tool call:

```jsonc
{
  "concepts": [
    {
      "concept": "Self-Attention",
      "summary": "A mechanism for weighting input token importance",
      "is_new": true,             // true if no existing wiki page covers this
      "tags": ["deep-learning", "nlp"],
      "confidence": 0.95,
      "provenance_state": "extracted",  // "extracted" | "merged" | "inferred" | "ambiguous"
      "contradicted_by": [               // slugs of concepts this one contradicts
        {"slug": "recurrent-processing", "reason": "Attention does not require recurrence"}
      ]
    }
  ]
}
```

The extraction prompt includes the list of **existing wiki page slugs**, so the LLM can avoid duplicating concepts. The source content is truncated to 100,000 characters per file.

Extractions happen in **batches** (default concurrency: 5 sources at a time), but each source is still processed sequentially within a batch (no parallel API calls).

After each batch, the state is saved to disk, so if the process crashes mid-way, only the current batch's progress is lost.

### Step 3: Merging by Slug

Multiple sources may mention the same concept. The merge step:

1. **Slugifies** each concept title (lowercase, kebab-case: `Self-Attention` → `self-attention`)
2. **Groups** concepts with the same slug
3. **Combines** source content for the page generation prompt
4. **Pessimistic merge**, takes the **minimum** confidence across sources (most conservative)
5. Sets `provenanceState` to `"merged"` when multiple sources contribute
6. Deduplicates `contradicted_by` entries

Example: if both `paper-1.md` and `paper-2.md` extract `attention`, they produce **one merged page** with `sources: [paper-1.md, paper-2.md]`.

### Step 4: Page Generation

For each merged concept, the LLM writes a wiki page:

> *"Write a comprehensive wiki page about '{concept}'. Use [[wikilinks]] to reference related pages."*

The prompt includes:
- The concept's title and summary
- Combined source content (up to 200,000 characters, fairly truncated if exceeded)
- A list of **related pages** (up to 5) to encourage cross-referencing
- The **existing page content** if this is an update (so the LLM can preserve manual edits)

Each page gets YAML frontmatter with `createdAt` preserved from the previous version (if any) and a fresh `updatedAt` timestamp.

### Step 5: Wikilink Resolution

After all pages are generated, the resolver does **deterministic regex-based** interlinking, no LLM calls:

```
Before:  "Attention is related to transformers."
After:   "Attention is related to [[transformer|Transformers]]."
```

**Two passes:**

1. **Outbound**, on newly written/changed pages, scan for mentions of other page titles
2. **Inbound**, on all existing pages, scan for mentions of brand-new concept titles

**Guard functions** prevent three types of damage:
- `_is_inside_wikilink`, won't nest wikilinks inside existing `[[...]]`
- `_is_inside_citation`, won't corrupt `^[source.md:5]` citation markers
- `_is_word_boundary`, won't link mid-word (e.g., "attention" in "inattentive" won't match)

The resolver is entirely regex-based, **no LLM calls**. This guarantees cross-references are complete and consistent, and takes milliseconds regardless of wiki size.

### Step 6: Orphan Detection & Frozen Slugs

**Orphan detection:**

A page is "orphaned" when **no other page links to it**. Orphaned pages:
- Get `orphaned: true` in their frontmatter
- Are excluded from the index
- Are excluded from query selection
- Remain on disk (never deleted, human review is always possible)

**Frozen slugs:**

When a source file is deleted, the compiler checks each concept the deleted source contributed to:
- **Exclusive concepts** (only mentioned in the deleted source) → page is **orphaned**
- **Shared concepts** (mentioned in other surviving sources) → the concept slug is **frozen**

Frozen slugs are preserved in their current state during the compile cycle. They are **unfrozen** once all owning sources have been successfully recompiled. This prevents data loss when a source is deleted mid-cycle.

A special case: if the LLM returns **zero concepts** for a source (failed extraction), the old concept list is preserved and the old slugs are temporarily frozen. The source will be retried on the next compile. This is the **empty hash retry trick**, the hash is saved as `""` so it's always detected as "changed" next time.

### Step 7: Index Generation

The compiler rebuilds `wiki/index.md`, a bullet-list table of contents:

```markdown
# Wiki Index

- [[self-attention|Self-Attention]]: A mechanism for weighting input token importance
- [[transformer|Transformer]]: A neural network architecture using only attention
- [[multi-head-attention|Multi-Head Attention]]: Parallel attention in subspaces

## Summary

- **Sources:** 10
- **Pages:** 49
- **Last compiled:** 2026-07-08T14:00:00Z
- **Duration:** 399000ms
```

Orphaned pages (`orphaned: true`) are excluded from the index.

### Step 8: Logging

Every compile appends an entry to `log.md`:

```markdown
## [2026-07-08] compile | 2 source(s) → 5 page(s) (12000ms)
- Sources: transformer-notes.md, attention-notes.md
- Created: [[self-attention]], [[transformer]]
- Updated: [[multi-head-attention]]
```

The log distinguishes **Created** (new pages) from **Updated** (existing pages modified). This gives you a readable history of what happened and when.

### Incremental Compile, How It Saves Time

On subsequent runs, unchanged sources are **skipped entirely**, no LLM calls, no page regeneration. Only new/changed sources are processed:

```
Compile 1 (10 sources): ~7 minutes (all 10 extracted, 49 pages generated)
Compile 2 (no changes):  ~0.1 seconds (nothing to do, early return)
Compile 3 (1 source edited): ~40 seconds (1 source re-extracted, updates pages)
```

**Affected sources** (cross-source dependency tracking): When a source changes, the compiler also checks if any **unchanged** sources shared concepts with it, those unaffected sources are also re-extracted in case their concept overlap needs updating. This is the `findAffectedSources()` / `findLateAffectedSources()` mechanism.

---

## 5. The Query Pipeline

### Three-Tier Page Selection

When you ask a question, the query pipeline tries three tiers in order:

```
Tier 1: Chunk-level embedding retrieval
  └─ Split pages into paragraphs, find chunks semantically similar to query
  └─ BM25 reranking for precision
  └─ Falls through if no embedding model available

Tier 2: Page-level embedding retrieval
  └─ Compare page-length vectors to query vector via cosine similarity
  └─ Falls through if no embedding model available

Tier 3: LLM reads wiki/index.md
  └─ LLM calls select_pages tool with the full index content
  └─ Returns up to 5 slugs
  └─ Fallback: first 3 pages if LLM fails
```

### Wikilink Graph Expansion

After pages are selected, the query does **wikilink graph expansion**:

```
Selected pages: [a20-nf-kappab-regulation, a20-apoptotic-resistance]
                        │
                        ▼
Parse each page for [[slug|Title]] patterns → find linked pages
                        │
                        ▼
Expanded pages: [a20-nf-kappab-regulation, a20-apoptotic-resistance,
                 a20-overexpression-in-gscs, a20-knockdown-effects]
                                                   ↑
                                    Pulled in via wikilinks (up to 3 extra pages)
```

This means even if the initial embedding/LLM selection misses a relevant page, it gets pulled in automatically if any selected page links to it. The expansion is limited to **one level deep** (no transitive link-following) and adds at most **3 extra pages**.

### Answer Generation

The selected (and expanded) pages are loaded and passed to the LLM:

```
Prompt: Question + Full content of selected pages (with [[wikilink]] markers)
  → LLM synthesizes answer with citations
```

The answer includes `[[wikilinks]]` to the cited pages so you can click through for detail.

### Saving Queries

With `--save`, the query result is written to `wiki/queries/` as a markdown file:

```markdown
# Query: How does A20 affect glioblastoma?

**Date:** 2026-07-08T09:18:05+00:00

## Selected Pages

- [[a20-and-nf-kappab-pathway-regulation]]
- [[a20-overexpression-in-glioblastoma-stem-cells]]

## Answer

A20 (TNFAIP3) functions as a **tumor enhancer in glioblastoma** by promoting
the survival of glioblastoma stem cells (GSCs) through NF-κB pathway
regulation [[a20-and-nf-kappab-pathway-regulation]]. It is overexpressed in
GSCs [[a20-overexpression-in-glioblastoma-stem-cells]]...
```

Saved queries are discoverable, future queries' index selection can find them, and the saved query page content becomes part of the wiki's knowledge.

---

## 6. Installation

### Requirements

- Python 3.10+
- `pip` (Python package installer)
- An LLM provider (see Configuration below)

### Install from Source

```bash
cd llm-wiki/
pip install -e .
```

This installs the `llm-wiki` CLI command and all required dependencies (`pyyaml`, `requests`).

### Verify Installation

```bash
llm-wiki --help
```

Or equivalently:

```bash
python -m llm_wiki.cli --help
```

Both produce:

```
usage: llm-wiki [-h] {compile,query,test} ...

LLM Wiki - compile and query

positional arguments:
  {compile,query,test}
    compile          Compile wiki from sources
    query            Query the wiki
    test             Run self-test
```

---

## 7. Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLMWIKI_PROVIDER` | `"anthropic"` | LLM provider: `"anthropic"` or `"openai"` |
| `LLMWIKI_MODEL` | `"claude-sonnet-4-20250514"` | Model name to use |
| `LLMWIKI_BASE_URL` | `"http://localhost:8000/v1"` | Base URL for OpenAI-compatible endpoints |
| `ANTHROPIC_API_KEY` | `""` | Required when using Anthropic provider |
| `OPENAI_API_KEY` | `"EMPTY"` | API key for OpenAI/vLLM endpoints |
| `LLMWIKI_PROMPT_BUDGET_CHARS` | `200000` | Max characters for combined source content in page gen |
| `LLMWIKI_EMBEDDING_MODEL` | `""` | Separate model for embeddings (optional) |
| `LLMWIKI_EMBEDDING_BASE_URL` | `""` | Separate endpoint for embeddings (optional) |

### Provider Setup

**Option A: Anthropic (cloud, default)**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export LLMWIKI_PROVIDER="anthropic"
export LLMWIKI_MODEL="claude-sonnet-4-20250514"
```

**Option B: OpenAI-compatible / vLLM (local)**

```bash
# Start vLLM with tool-call support:
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --port 8000 \
  --api-key token-abc123 \
  --enable-auto-tool-choice \
  --tool-call-parser openai

# Configure the CLI:
export LLMWIKI_PROVIDER="openai"
export LLMWIKI_MODEL="Qwen/Qwen3-14B"
export LLMWIKI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="token-abc123"
```

> **Important**: The OpenAI-compatible provider (used for vLLM, Ollama, etc.) requires the model to support **tool/function calling**. This is needed for concept extraction and page selection. The Anthropic provider supports this natively.

### Provider Differences

| Feature | Anthropic | OpenAI-compatible |
|---|---|---|
| Tool calling | Native via Messages API | Requires model support (e.g., Qwen3, GPT) |
| Think-tag stripping | Not needed | Automatic (strips `<think>` blocks from reasoning models) |
| API format | Anthropic Messages | OpenAI Chat Completions |
| Auth | `x-api-key` header | `Bearer` header |

---

## 8. Usage Guide

### 8.1 Quick Start (Self-Test)

The fastest way to see LLM Wiki in action:

```bash
# With Anthropic:
export ANTHROPIC_API_KEY="sk-ant-..."
llm-wiki test

# With local vLLM:
export LLMWIKI_PROVIDER="openai"
export LLMWIKI_MODEL="Qwen/Qwen3-14B"
export LLMWIKI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="token-abc123"
llm-wiki test
```

This creates a temporary directory, writes 2 test source files, compiles them, verifies the output, runs a no-op compile (confirming incremental detection), runs an incremental compile (editing one source), and cleans up. Expect to see:

```
Compile 1...
  Changed: 5, Created: 5, Duration: 53606ms
  Pages: 5
    machine-learning.md: title=Machine Learning, sources=['test1.md']
    deep-learning.md: title=Deep Learning, sources=['test2.md']
    ...
  Index: OK
  State: OK
  No-op compile: OK
  Incremental compile: OK (2 pages changed, 0 created)
All tests passed!
```

### 8.2 Build Your Own Wiki

**Step 1: Create source files**

```bash
mkdir -p my-wiki/sources
cd my-wiki

# Add your research notes as .md files
echo "# Attention Is All You Need

The Transformer architecture relies entirely on self-attention...
" > sources/attention-paper-summary.md

echo "# Deep Learning Basics

Deep learning uses neural networks with multiple layers...
" > sources/deep-learning-notes.md
```

Sources can be paper summaries, meeting notes, transcripts, analysis, any markdown text. Files should be at least 50 characters. Each source should be a single `.md` file with a unique filename.

**Step 2: Compile**

```bash
llm-wiki compile
# or: llm-wiki compile --dir /path/to/my-wiki
```

The compiler will:
1. Detect new sources (hash comparison against `state.json`)
2. Extract 3–8 concepts from each source via LLM tool calls
3. Merge concepts with identical slugs, combining source content
4. Generate a wiki page for each concept with frontmatter and wikilinks
5. Resolve `[[wikilinks]]` deterministically (regex-based, no LLM needed)
6. Detect orphaned pages (no incoming links → `orphaned: true`)
7. Build `wiki/index.md` with all page summaries
8. Append to `log.md`
9. Save new hashes to `.llmwiki/state.json`

```json
{
  "changed": ["self-attention", "transformer", ...],
  "created": ["self-attention"],
  "updated": ["transformer"],
  "deleted": [],
  "frozen": [],
  "pageCount": 5,
  "sourceCount": 2,
  "durationMs": 12000
}
```

**Step 3: Query**

```bash
llm-wiki query "How does attention work in transformers?"
# or: llm-wiki query "What is DIC?" --dir /path/to/my-wiki
# or: llm-wiki query "What is DIC?" --save   (saves to wiki/queries/)
```

The query pipeline:
1. Tries chunk embedding → page embedding → LLM index reading (tiered fallback)
2. Expands selected pages via wikilink graph (follows `[[links]]` one level deep)
3. Loads expanded page content and passes to LLM for answer synthesis

**Step 4: Add more sources and recompile**

```bash
# Add a new source
echo "# New Research Notes..." > sources/new-paper.md

# Incremental compile, only processes new/changed sources
llm-wiki compile
```

Incremental compile is fast because unchanged sources are cached in `state.json`.

### 8.3 Controlling Concurrency

```bash
llm-wiki compile --concurrency 10   # Extract 10 sources per batch
```

Default concurrency is 5. Higher values process more sources per batch but hit the LLM API harder (more concurrent requests).

---

## 9. Module Reference

The code is organized into 10 Python modules in the `llm_wiki/` package:

### `types.py`, Data Classes

All data types used throughout the pipeline:

| Type | Fields | Purpose |
|---|---|---|
| `SourceState` | `hash`, `concepts`, `compiledAt` | Per-source persisted state |
| `WikiState` | `version`, `sources`, `frozenSlugs`, `indexHash` | Top-level state from `state.json` |
| `ExtractedConcept` | `concept`, `summary`, `is_new`, `tags`, `confidence` | Raw LLM extraction output |
| `MergedConcept` | `slug`, `title`, `summary`, `sourceFiles`, `combinedContent` | Post-merge, pre-generation |
| `Frontmatter` | `title`, `summary`, `sources`, `kind`, `createdAt`, ... | YAML frontmatter for wiki pages |
| `CompileResult` | `changed`, `created`, `updated`, `pageCount`, `sourceCount`, `durationMs` | Compile output |
| `QueryResult` | `question`, `selectedPages`, `answer`, `saved` | Query output |
| `LLMTool` | `name`, `description`, `input_schema` | Tool definition for LLM tool calling |

### `constants.py`, Configuration Constants

| Constant | Default | Description |
|---|---|---|
| `SOURCES_DIR` | `"sources"` | Where source files live |
| `CONCEPTS_DIR` | `"wiki/concepts"` | Where generated pages go |
| `QUERIES_DIR` | `"wiki/queries"` | Where saved query results go |
| `STATE_FILE` | `".llmwiki/state.json"` | Persisted compile state |
| `LOCK_FILE` | `".llmwiki/lock"` | File lock for crash safety |
| `INDEX_FILE` | `"wiki/index.md"` | Generated table of contents |
| `MAX_SOURCE_CHARS` | `100_000` | Source content truncation limit |
| `MIN_SOURCE_CHARS` | `50` | Minimum source content length |
| `DEFAULT_PROMPT_BUDGET_CHARS` | `200_000` | Combined source budget for page gen |
| `QUERY_PAGE_LIMIT` | `5` | Max pages selected per query |
| `MAX_CONCEPTS` | `8` | Max concepts per source extraction |
| `RETRY_COUNT` | `3` | LLM API call retries |
| `RETRY_BASE_MS` | `1000` | Base delay for retry backoff |
| `RETRY_MULTIPLIER` | `4` | Exponential factor for retry backoff (1s → 4s → 16s) |

### `llm.py`, LLM Provider Abstraction

Two providers, one interface:

```
BaseProvider
  ├── complete(system, prompt) → str           // Freeform text generation
  ├── extract_concepts(system, prompt) → list   // Structured tool-call extraction
  ├── select_pages(system, prompt, max_items)   // Tool-call page selection
  └── embed(text, input_type) → list[float]    // Embedding (optional)

AnthropicProvider
  └── Sends requests to api.anthropic.com (Messages API)

OpenAIProvider
  └── Sends requests to configurable base URL (OpenAI Chat Completions)
```

Key details:
- Both providers support **exponential backoff with jitter** for retries
- Both providers support **tool calling** (Anthropic uses native tools; OpenAI uses `tool_choice` with function definitions)
- Both providers implement `select_pages` as a tool call (not JSON completion), with a fallback to parsing JSON from freeform completion
- The `OpenAIProvider` automatically strips `<think>` blocks from responses (reasoning models like Qwen3 wrap responses in these)
- The `OpenAIProvider` sends embedding requests to a separate configurable endpoint (`LLMWIKI_EMBEDDING_BASE_URL`)
- 4xx errors (except 429) are **not retried**, they indicate client errors that won't succeed on retry

Provider selection:

```python
from llm_wiki.llm import get_provider
provider = get_provider()  # Reads LLMWIKI_PROVIDER env var
```

### `compiler.py`, Main Compile Pipeline

The orchestrator. Calls all other modules in sequence:

```python
compile_wiki(root) → CompileResult
```

**Workflow:**

1. `ensure_dirs()`, create missing directories
2. `acquire_lock()`, prevent concurrent compiles
3. `load_state()`, read previous state from `.llmwiki/state.json`
4. `_load_sources()`, scan `sources/` for `.md` files (≥50 chars)
5. `_detect_changes()`, SHA-256 comparison against state
6. `findAffectedSources()`, cross-source dependency detection (unchanged sources that share concepts with changed ones)
7. **Extraction phase**: `_extract_concepts_batch()`, LLM tool call per source, with per-source state persistence for crash safety
8. **Late-affected expansion**: `findLateAffectedSources()`, post-extraction check for newly emerging concept overlaps
9. `_merge_concepts()`, deduplicate by slug, pessimistic confidence merge
10. `_generate_page()`, LLM writes each wiki page with existing content + related pages context
11. **Orphan marking**, mark exclusive-concept pages as `orphaned: true` when source deleted
12. **Frozen slug persistence**, unfreeze slugs when all owners recompiled
13. `resolve_links()`, two-pass deterministic wikilink resolution
14. `generate_index()`, rebuild `wiki/index.md`
15. `_refresh_embeddings()`, update embedding vectors (if provider supports it)
16. `save_state()`, persist new hashes to `.llmwiki/state.json`
17. `append_log()`, append to `log.md`
18. `release_lock()`, always runs (finally block)

### `query.py`, Question Answering

Three-tier query pipeline:

```python
query(root, question, save=False) → QueryResult
```

**Workflow:**

1. `_load_wiki_pages()`, reads all non-orphaned pages from `wiki/concepts/`
2. **Tier 1**: Chunk-level embedding retrieval (BM25 reranking)
3. **Tier 2**: Page-level embedding retrieval (cosine similarity)
4. **Tier 3**: LLM reads `wiki/index.md` via `select_pages` tool call
5. **Wikilink expansion**: `_expand_wikilinks()`, follows `[[links]]` on selected pages (up to 3 extra pages)
6. `_load_selected_pages()`, loads content of selected + expanded pages
7. LLM synthesizes answer with `[[wikilinks]]` citations
8. Optional: saves query result to `wiki/queries/`
9. Lightweight index refresh so saved query is discoverable
10. Append query to `log.md`

### `utils.py`, Infrastructure Utilities

| Function | Purpose |
|---|---|
| `slugify(text)` | Convert text to kebab-case filename slug (max 60 chars) |
| `sha256(text)` | Compute SHA-256 hex digest |
| `ensure_dirs(root)` | Create wiki directories (`wiki/concepts/`, `wiki/queries/`, `.llmwiki/`) |
| `now_iso()` | Current UTC timestamp in ISO format (seconds precision) |
| `numbered_lines(text)` | Right-aligned line numbers for source content citations |
| `build_concept_to_sources_map(state)` | Reverse index: concept slug → list of source filenames |
| `format_source_sections(sources, budget)` | Combine source content with fair-share prompt budget truncation |
| `parse_frontmatter(text)` | Parse YAML frontmatter from page content → `(Frontmatter, body)` |
| `format_frontmatter(fm)` | Build YAML frontmatter string from `Frontmatter` object |
| `atomic_write(path, content)` | Safe file write via temp file + rename (no partial writes) |
| `read_maybe(path)` | Read file or return `""` (never raises) |
| `load_state(root) / save_state(root, state)` | Read/write `state.json` with `.bak` corruption recovery |
| `acquire_lock(root) / release_lock(root)` | File-based `O_EXCL` mutex with 30s timeout |
| `append_log(root, entry)` | Append to `log.md` with created vs updated distinction |
| `generate_index(root, ...)` | Build `wiki/index.md` excluding orphaned pages |
| `select_related(entries, current_slug, max_count)` | Pick related pages for cross-referencing |
| `with_retry(fn)` | Exponential backoff retry wrapper |

### `prompts.py`, LLM Prompt Templates

Contains the exact system and user prompts for the four LLM interactions:

| Template | Used For | Key Instructions |
|---|---|---|
| `EXTRACTION_SYSTEM` / `EXTRACTION_PROMPT` | Concept extraction (tool call) | Extract 3-8 concepts, dedup against existing pages, set provenance state |
| `PAGE_GEN_SYSTEM` / `PAGE_GEN_PROMPT` | Wiki page generation (freeform) | Write encyclopedic entries, use `[[wikilinks]]`, cite with `^[file.md:N-M]` |
| `QUERY_SELECTION_SYSTEM` / `QUERY_SELECTION_PROMPT` | Page selection during query | Select up to 5 most relevant pages from index |
| `QUERY_ANSWER_SYSTEM` / `QUERY_ANSWER_PROMPT` | Answer generation during query | Answer from wiki content only, cite with `[[wikilinks]]` |

### `resolver.py`, Wikilink Resolution

Deterministic interlink resolution (no LLM calls):

| Function | Purpose |
|---|---|
| `resolve_links(root, concepts_dir, changed_slugs, new_slugs)` | Two-pass outbound + inbound wikilink resolution |
| `_resolve_outbound(body, page_slug, title_index)` | Scan page body for mentions of other page titles |
| `_resolve_inbound(body, new_title, new_slug)` | Scan page body for mentions of a new concept title |
| `build_title_index(root, concepts_dir)` | Map slug → title for all non-orphaned pages |
| `resolve_wikilinks(text, slug, known_slugs)` | Convert `[[target|display]]` syntax in LLM-generated text |
| `get_incoming_links(all_pages, target_slug)` | Find all pages that link to a specific slug |
| `update_orphan_status(root, concepts_dir)` | Mark pages with 0 incoming links as orphaned (interlink-based) |
| `orphan_page(root, slug, concepts_dir)` | Surgical insert of `orphaned: true` in frontmatter (source-based) |

Guard functions:
- `_is_inside_wikilink(text, pos)`, prevents nesting wikilinks
- `_is_inside_citation(text, pos)`, prevents corrupting citation markers
- `_is_word_boundary(text, start, end)`, prevents mid-word linking

### `embeddings.py`, Embedding Store (Optional Tier)

Used only when an embedding model is available:

| Function/Class | Purpose |
|---|---|
| `EmbeddingStore(root)` | Load/save page-level + chunk-level embedding vectors to `.llmwiki/embeddings.json` |
| `split_into_chunks(body)` | Paragraph-aligned chunking (target 800 chars, max 1400, min 200) |
| `cosine_similarity(a, b)` | Vector similarity for ranking |
| `BM25Ranker(candidates)` | BM25 reranking over chunk-level candidates |
| `find_top_k_pages(query_vec, k)` | Page-level cosine similarity search |
| `find_top_k_chunks(query_vec, k)` | Chunk-level cosine similarity search |

The embedding store is **non-blocking**, any error during embedding is silently caught. The query pipeline gracefully degrades to tier 3 (LLM reads index) when embeddings are unavailable.

### `cli.py`, Command-Line Interface

Entry point for the `llm-wiki` command and the built-in self-test:

```bash
llm-wiki compile          # Compile wiki from sources/
llm-wiki query "..."      # Answer a question
llm-wiki test             # Run self-test
```

The self-test (`_run_test()`) creates a temp wiki, compiles 2 sources, verifies pages/index/state, runs a no-op compile, then an incremental compile. It uses `OpenAIProvider` (so it can point at a local model) but is provider-agnostic.

---

## 10. Edge Cases & Important Details

### 10.1 Empty Sources Directory

If `sources/` doesn't exist or is empty, the compiler prints "No changes detected" and exits. No error, just nothing to do.

### 10.2 Short Sources (< 50 chars)

Sources shorter than 50 characters are silently skipped. They don't have enough content for meaningful concept extraction.

### 10.3 Large Sources (> 100K chars)

Content is truncated to 100,000 characters per source. A warning is printed during extraction. The `MAX_SOURCE_CHARS` constant can be modified if needed.

### 10.4 Failed Extraction (0 Concepts)

If the LLM returns no concepts for a source, the extraction result is just empty, no page is created. The source hash is saved as `""` (empty string) so the next compile will detect it as "changed" and retry. Old concept slugs from this source are **frozen** during the retry cycle to prevent data loss.

### 10.5 No-Op Compile

If no sources changed since the last compile, the pipeline exits immediately after refreshing the index. **No LLM calls are made.** The log still records the no-op with 0 changes.

### 10.6 Incremental Compile

When a source is edited, only that source is re-extracted. New pages are generated for new/changed concepts. Unchanged sources are untouched. The affected-sources mechanism also catches unchanged sources that share concepts with changed ones.

### 10.7 Source Deletion

When a source file is deleted:
- **Exclusive concepts** (only mentioned in the deleted source) → page is marked `orphaned: true` but stays on disk
- **Shared concepts** (mentioned in other surviving sources) → concept slug is **frozen** during compile, then unfrozen when all owners recompile

### 10.8 Slug Collisions

If two different concepts produce the same slug (e.g., "ML" and "Machine Learning" both slugify to `"machine-learning"`), the merge step groups them into one page. The combined content from all sources is presented to the LLM for page generation.

### 10.9 Empty Slug from `slugify()`

If a concept title contains only special characters or emoji, `slugify()` returns `""`. This would create a hidden file `.md`. The compiler guards against this, empty slugs are skipped with a warning. (In practice, this is rare since the LLM generates meaningful titles.)

### 10.10 Crash During Compile

If the compile is interrupted mid-way:
- **Pages already written** → they stay on disk (valid but may be incomplete)
- **State partially saved** → per-source state is saved after each extraction batch; at most one batch of progress is lost
- **Lock file stays** → the lock is in `.llmwiki/lock`; delete it manually if the process truly crashed
- **Partial state recovery** → re-running compile will re-extract sources whose hashes weren't saved

### 10.11 Corrupt `state.json`

If `state.json` is malformed (JSON decode error), `load_state()`:
1. Copies the corrupt file to `state.json.bak` for manual recovery
2. Returns an empty state (version 1, no sources)
3. The next compile treats everything as new, a full recompile

### 10.12 LLM API Failures

If an LLM call fails (network error, API error, timeout):
- `with_retry` retries up to `RETRY_COUNT` (3) times with exponential backoff: 1s → 4s → 16s
- **Jitter** (random delay component) prevents thundering herd on retry
- 4xx errors (except 429 "Too Many Requests") are **not retried**, they indicate permanent client errors
- After all retries fail, the extraction for that source is **skipped with a warning**
- The source will be retried on the next compile (hash not saved)

### 10.13 Concurrency for Large Wikis

The extraction loop processes sources in batches, configurable via `--concurrency` (default 5, `DEFAULT_COMPILE_CONCURRENCY`). Each source within a batch is extracted sequentially. The batch approach provides a simple concurrency throttle without parallel API calls.

### 10.14 Log Corruption Recovery

If `log.md` is deleted or corrupted, `append_log()` will re-create it with a fresh header. Past entries are lost but new entries accumulate from that point.

---

## 11. What's Not Included

This is a **minimal** implementation. The following features from reference systems are intentionally excluded:

| Feature | Why Excluded |
|---|---|
| **Multi-format ingest** (PDF, Office, web) | Separate concern; user provides `.md` files manually |
| **Concurrent LLM calls** | Serial extraction is simpler and avoids rate limits |
| **Review/lint system** (review policy gate) | User review of generated pages is sufficient for personal use |
| **Database backend** (SQLite, Supabase) | Filesystem + `state.json` is sufficient for individual use |
| **Obsidian-specific metadata** | Vendor lock-in |
| **MCP server** | Infrastructure concern; minimal version is CLI-only |
| **Automatic cascade updates** | Replaced by deterministic interlink resolution |
| **Output language config** | Not implemented (could be added via prompt injection) |
| **Web UI / graph viewer** | CLI-only; pages are plain markdown viewable in any editor |
| **Chrome extension / web clipper** | Social feature; manual file creation is sufficient |

---

## 12. Comparison with Other LLM Wikis

A side-by-side comparison across 5 reference implementations and our minimal Python version.

### Page Creation (How wiki pages are made)

| Dimension | Karpathy Gist (idea) | karpathy-llm-wiki (Astro-Han skill) | llm-wiki-compiler (TS) | llmwiki MCP (Python) | Hermes Agent (skill) | **This project** |
|---|---|---|---|---|---|---|
| **Creates pages via** | LLM inline during ingest | LLM writes inline during ingest | LLM tool-call pipeline (extract→merge→generate) | LLM writes via MCP `create`/`edit`/`append` tools | LLM writes inline during ingest | LLM tool-call pipeline (extract→merge→generate) |
| **Automatic compilation** | Yes (as part of ingest) | Yes (as part of ingest) | Yes (`llmwiki compile` with change detection) | No (relies on Claude Routines/scheduled prompts) | Yes (as part of ingest) | Yes (`llm-wiki compile` with SHA-256 change detection) |
| **Source format** | Any (web/file/paste) | Any (web/file/paste) | Any (web, PDF, image, file) via `ingest` command | Any (PDF, Office, web, text, image) via file watcher | Any (web/file/paste) | Only `.md` files in `sources/` |
| **Multi-source merge** | LLM decides inline | LLM decides inline | Slug-based merge with pessimistic confidence | Not automated (LLM reads both sources) | LLM decides inline | Slug-based merge with pessimistic confidence |
| **Page kinds** | Single type | Single type | Typed (concept, entity, comparison, overview) | Single type | Typed (entities, concepts, comparisons) | Single type (concept) |
| **Human review gate** | None | Heuristic lint | Policy-engine (candidates/ folder, approve/reject) | MCP `lint` tool | Heuristic lint | None |
| **Seed pages** | None | None | Schema-declared, materialized on each compile | None | None | None |

### Query / Retrieval (How questions are answered)

| Dimension | Karpathy Gist | karpathy-llm-wiki skill | llm-wiki-compiler (TS) | llmwiki MCP (Python) | Hermes Agent skill | **This project** |
|---|---|---|---|---|---|---|
| **Primary retrieval** | LLM reads index.md | LLM reads index.md | Chunk-level embeddings + BM25 rerank | SQLite FTS5 full-text keyword search | LLM reads index.md | Chunk-level embeddings + BM25 rerank |
| **Fallback method** | None | None | Page-level embeddings → LLM reads index.md | None | None | Page-level embeddings → LLM reads index.md |
| **What LLM sees** | Full page bodies for selected pages | Full page bodies for selected pages | Full page bodies + top chunk excerpts for selected pages | Search result snippets → full page via `read` tool | Full page bodies for selected pages | Full page bodies for selected pages (NO chunk excerpts) |
| **Embedding vectors** | No | No | Yes (Voyage/Ollama/OpenAI), JSON store | No | No | Yes (OpenAI-compatible only), JSON store |
| **Chunk-level search** | No | No | Yes (paragraph-aligned, content-hash dedup) | Yes (~512-token, ~128-overlap, header-aware) | No | Yes (paragraph-aligned, BM25 rerank) |
| **Graph expansion** | None | None | None (MCP context pack tool) | Reference graph (backlinks, staleness propagation) | None | Wikilink graph expansion (1 level, +3 pages) |
| **Saved queries** | Yes (as wiki pages) | Yes (Archive feature) | Yes (wiki/queries/) | No | Yes (queries/ directory) | Yes (wiki/queries/ with --save flag) |

### Architecture

| Dimension | Karpathy Gist | karpathy-llm-wiki skill | llm-wiki-compiler (TS) | llmwiki MCP (Python) | Hermes Agent skill | **This project** |
|---|---|---|---|---|---|---|
| **Nature** | Pattern description | Agent SKILL.md (prompt-only) | TypeScript CLI + SDK + MCP server | Python MCP server + FastAPI + Chrome extension | Agent SKILL.md (prompt-only) | Python CLI tool |
| **Storage** | Filesystem | Filesystem | Filesystem + JSON state files | SQLite/Postgres + filesystem | Filesystem | Filesystem + JSON state files |
| **Search index** | index.md only | index.md only | embeddings.json + index.md | FTS5 virtual table | index.md only | embeddings.json + index.md |
| **LLM ownership** | Full (LLM writes wiki, human reads) | Full (LLM writes wiki, human reads) | Full (LLM writes wiki, compiler orchestrates) | Delegated (LLM decides what to write via tools) | Full (LLM writes wiki, human reads) | Full (LLM writes wiki, pipeline orchestrates) |
| **Locking** | None | None | File-based O_EXCL with two-phase stale reclamation | Database transaction | None | File-based O_EXCL with 30s timeout |
| **Concurrency** | N/A (LLM decides) | N/A (LLM decides) | Batch processing with pLimit | Background tasks (watchfiles) | N/A (LLM decides) | Batch processing (serial per batch) |

### Data Model

| Dimension | Karpathy Gist | karpathy-llm-wiki skill | llm-wiki-compiler (TS) | llmwiki MCP (Python) | Hermes Agent skill | **This project** |
|---|---|---|---|---|---|---|
| **Page format** | Markdown with `> Source:` header | Markdown with `## Sources` table | Markdown + YAML frontmatter | Markdown + YAML frontmatter | Markdown with `## Sources` table | Markdown + YAML frontmatter |
| **Metadata** | Minimal (`> Source:`, `> Raw:`) | Minimal (`Source`, `Raw` fields) | Rich (title, sources, summary, kind, createdAt, updatedAt, confidence, provenanceState, contradictedBy, tags, orphaned) | Rich (title, description, date, tags via MCP auto-frontmatter) | Rich (tag taxonomy, page thresholds) | Rich (title, summary, sources, kind, createdAt, updatedAt, confidence, provenanceState, tags, orphaned) |
| **Chunk format** | None | None | JSON ({slug, title, chunkIndex, contentHash, text, vector}) | SQLite row ({content, header_breadcrumb, page, token_count}) + FTS5 | None | JSON ({slug, title, chunkIndex, text, vector}) in embeddings.json |
| **State persistence** | None | log.md | state.json (per-source hash, concepts, compiledAt) | SQLite document_chunks table | log.md | state.json (per-source hash, concepts, compiledAt, frozenSlugs) |
| **Citation format** | Markdown links | Markdown links | `^[filename.md:START-END]` + frontmatter `sources` array | Footnote numbers → filename in references graph | Markdown links | `^[filename.md:START-END]` + frontmatter `sources` array |

### Differences vs This Project (Notable)

| Reference | Key difference from our approach |
|---|---|
| **Karpathy Gist** | Pure idea description; no code, no automation. Our project is an executable implementation of his vision with formal pipeline stages and change detection. |
| **karpathy-llm-wiki skill** | No embeddings, no deterministic link resolution — all LLM-driven. No chunking, no query tiers. Our project adds embeddings, wikilink graph expansion, and a 3-tier query pipeline. |
| **llm-wiki-compiler (TS)** | Most similar architecture (we share the same design). Differences: TS has a review/policy gate (candidates/), concurrent extraction with pLimit, 5+ providers, typed page kinds (concept/entity/comparison), eval harness, OKF export, two-phase stale lock. Our Python version is simpler: no review gate, no page kinds, fewer providers, no eval. |
| **llmwiki MCP (Python)** | Not a compiler — it's an infrastructure layer (MCP tools + API + auto file watcher). The LLM is expected to write pages through tools rather than a pipeline. Has SQLite/Postgres, FTS5 search, Chrome extension, PDF/Office extraction, hosted multi-user mode, reference graph with staleness propagation. Our project is compiler-first (batch compile → durable wiki), not tool-first. |
| **Hermes Agent** | Has SCHEMA.md with mandatory tag taxonomy (prevents tag sprawl), explicit page thresholds (merge at 50% overlap, split at 10 sections/300 lines), session continuity protocol (read SCHEMA + index + log on every session), scaling rules (50/200/500 entry limits). Our project has none of these — relies entirely on LLM judgment. |

### Summary

| Aspect | The 5 references collectively do | This project does differently |
|---|---|---|
| **Ingestion** | Accept any format, auto-convert | Only `.md` files in `sources/` |
| **Page quality** | Have review gates, lint, page typing, thresholds | No gate, single page kind, LLM-judgment-only |
| **Query precision** | Have chunk-feeding, reference graphs, FTS5 | Full-page feeding, no chunk context for LLM |
| **Infrastructure** | Multiple have DB backends, hosted mode, MCP, web UI | Filesystem-only, CLI-only |
| **Persistence** | Some have SQLite FTS, citation graph, staleness tracking | JSON state, source-based orphan, frozen slugs |

This project starts as a minimal viable compiler. The references show what can be added: review gates, chunk-level query feeding, reference graphs, auto-ingest for non-text formats, and DB-backed scalability.
