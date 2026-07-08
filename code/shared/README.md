# shared/, canonical library modules used across stages

Flat (no subdirs) so `sys.path.insert(0, ".../code")` + `from shared import stats` (or
`import stats`, `import data as D`, etc., if `code/shared` itself is put on `sys.path`)
works without package-relative import gymnastics.

These six files are byte-identical copies of the certification-contract library that
`code/certify/` builds (see `code/certify/README.md`), re-exported here because they are
imported by other stages' code (`code/maintain/`, `code/retract/`):

| file | source (canonical) | what's imported from it |
|---|---|---|
| `oai_client.py` | `code/certify/contract/oai_client.py` | `VLLM`, chat-completion client factory (HTTP vLLM server or in-process offline engine via `CONTRACT_OFFLINE_MODEL`) |
| `maintain.py` | `code/certify/contract/maintain.py` | `Store`, `PageRouter`, compiled-store object with page-level provenance/write-tracking, and nearest-page routing for incoming docs |
| `certify.py` | `code/certify/contract/certify.py` | `CLEAN_SYS`, `JUDGE_SYS`, `doc_text`, `split_claims`, `assign_pages`, `judge`, `judge_many`, compiler/judge prompts and claim-judging helpers |
| `stats.py` | `code/certify/contract/stats.py` | `cp_lower`, `cp_upper`, `retention_lcb`, `corrected_point`, Clopper–Pearson bounds and the judge-noise correction |
| `currency.py` | `code/certify/contract/currency.py` | `CLEAN_SYS`, `INCR_SYS`, `ANS_SYS`, `match`, `assign_pages`, `docs_text`, currency/supersession prompts and answer-matching |
| `data.py` | `code/certify/harness/data.py` | `load_jsonl`, `load_corpus`, `load_claims`, `gold_doc_ids`, `subsample_corpus` (imported as `import data as D`), SciFact-Open dataset loaders |

If you edit the certification logic, edit `code/certify/contract/` (or `harness/`) and
copy the change here, these are duplicated on purpose (small text files, cross-stage
import convenience) rather than symlinked.
