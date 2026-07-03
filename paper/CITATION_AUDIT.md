# Citation Audit Report

**Date**: 2026-07-03
**Bib file**: `references.bib`
**Total entries**: 110
**Method**: each flagged entry independently web-verified against arXiv/DBLP/publisher sources

## Summary

A pre-publication pass over the bibliography flagged 8 entries as structurally suspicious — internally inconsistent year/arXiv-ID metadata, a lone `author={Anonymous}` entry, one likely author-surname transcription error, and a citation-key collision on this paper's foundational premise citation (Karpathy's LLM-Wiki post). All 8 were independently verified against arXiv/DBLP/publisher sources.

## Findings

| Key | Verdict | Issue | Fix applied |
|---|---|---|---|
| `packb` | FIX | Real arXiv ID (2507.13550), but title/author were fabricated — no "PAC" framing exists in the real paper | Corrected to real title/authors (Garrido-Merchán & Puente) |
| `shi2024crag` | FIX | Wrong first-author surname | Corrected to Mintong Kang (ICML 2024, PMLR v235) + added arXiv:2402.03181 |
| `li2024debuglm` | FIX | Year field said 2024, arXiv ID implies 2026 (2603.17884); topic mismatch (real paper is about provenance tracing, not RAG causal attribution) | Year corrected to 2026; noted topic drift so it's not miscited |
| `debenedetti2024eraser4rag` | FIX | Year said 2024, arXiv ID implies 2025 (2504.09910) | Year corrected to 2025; topical match confirmed real |
| `chen2024bloomscrub` | FIX | Year said 2024, arXiv ID implies 2025 (2504.16046); real paper is about copyright-infringement mitigation, NOT RAG knowledge certification | Year corrected to 2025; flagged as likely **not relevant** to this paper's related work — do not cite for RAG-certification claims |
| `ye2024harrypotterleak` | FIX | Year said 2024, arXiv ID implies 2025 (2505.17160); confirmed distinct from Eldan & Russinovich's 2023 "Who's Harry Potter?" | Year corrected to 2025 |
| `karpathy2026wiki` | KEEP | Real gist, confirmed at `gist.github.com/karpathy/442a6bf555914893e9891c11519de94f` | No change — this is the paper's foundational premise citation |
| `karpathy2026companywiki` | REMOVE | Fabricated/duplicate — the claimed URL resolves to Karpathy's generic homepage, no distinct "company wiki" post exists | **Deleted.** Only one real Karpathy LLM-Wiki source exists; use `karpathy2026wiki` exclusively |

**Net**: 111 → 110 entries. Five entries had internally-inconsistent year/arXiv-ID pairs corrected — a real arXiv ID with a fabricated or drifted title/topic/year is a specific, previously-seen failure mode, worth a dedicated verification pass rather than assuming the bibliography was clean.

## Action for paper writing

`chen2024bloomscrub` and `li2024debuglm` are real papers but topically adjacent, not on-point — do not cite them for claims their titles used to (falsely) suggest. Prefer not citing them at all unless a genuinely accurate use is found; an uncited-but-correct bib entry is harmless, a wrong-context citation is not.

## Not yet run

This audit covered only the 8 structurally-flagged entries, not all 110 (no paper text existed yet to extract citation contexts from at the time). Before further revision, a context-appropriateness pass over every `\cite{}` against its surrounding sentence in the finished `main.tex` would be the natural next step — this was intentionally deferred rather than run against placeholder text.
