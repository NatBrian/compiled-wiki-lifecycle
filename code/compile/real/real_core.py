"""Shared core for the real-published-system baselines (LightRAG, Mem0, RAPTOR,
GraphRAG, Graphiti).

Design for a clean, credible comparison:
  * Each real system does its OWN indexing + retrieval over the SAME anonymized
    per-release document stream our arms see (no extra signal).
  * We then hold the READER CONSTANT: the system's retrieved context is passed
    through the SAME terse reader prompt + SAME vLLM (Qwen2.5-14B) our arms use,
    and scored by the SAME token-match scorer. This isolates the one variable
    under test -- the retrieval representation (accumulate vs resolve) -- from
    each system's generation verbosity.

Pure-python (corpus IO + scoring); the heavy system libs live in per-system venvs
and import this module via sys.path.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ----- corpus -----------------------------------------------------------------

def anon_doc_text(d: dict) -> str:
    """Identical to arms.doc_text under BENCH_ANON=1: no version, no supersedes."""
    return f"{d['symbol']}: to {d['concept']}."


def load_corpus(lib: str):
    c = ROOT / "corpus" / lib
    versions = json.loads((c / "manifest.json").read_text())["versions"]
    bundles, seq = {}, 0
    for v in versions:
        p = c / f"v{v}.jsonl"
        docs = []
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                d["_text"] = anon_doc_text(d)
                d["ingest_seq"] = seq
                d["cid"] = d["id"].split("__")[0]
                seq += 1
                docs.append(d)
        bundles[v] = docs
    gold = [json.loads(l) for l in (c / "gold.jsonl").read_text().splitlines() if l.strip()]
    by_version = defaultdict(list)
    for q in gold:
        by_version[q["ask_at_version"]].append(q)
    return versions, bundles, by_version


# ----- reader (constant across all systems) -----------------------------------

_client = None

def read_answer(context: str, query_text: str, *, synthesis: bool = False,
                model=None, url=None) -> dict:
    """The SAME terse reader our arms use (arms._read), against our vLLM."""
    global _client
    from openai import OpenAI
    model = model or os.environ.get("AGENT_MODEL", "qwen2.5-14b")
    url = url or os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
    if _client is None:
        _client = OpenAI(base_url=url, api_key="dummy")
    if synthesis:
        system = ("You are a precise coding assistant. Use ONLY the context. "
                  "Concisely explain how the API changed; name old and current symbols.")
        max_tokens = 300
    else:
        system = ("You are a precise coding assistant. Use ONLY the context. "
                  "Answer with the exact API call or symbol only, nothing else.")
        max_tokens = 40
    r = _client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}])
    u = r.usage
    return {"answer": r.choices[0].message.content or "",
            "qtokens": (u.prompt_tokens + u.completion_tokens) if u else 0}


# ----- scoring (identical to src/scorer.py factual path) ----------------------

STOP = {"model", "class", "field", "data", "true", "false", "self", "def",
        "import", "from", "none", "the", "to", "a", "in", "is", "of", "way",
        "correct", "exact", "api", "call", "symbol", "with"}

def _tokens(s): return set(re.findall(r"[a-z_][a-z0-9_]+", s.lower()))
def _keys(sym):
    k = _tokens(sym) - STOP
    return k or _tokens(sym)

def score_factual(gold, deprecated, answer):
    ans = _tokens(answer)
    correct = bool(_keys(gold) & ans)
    stale = False
    if not correct:
        for d in deprecated:
            if _keys(d) & ans:
                stale = True
                break
    return {"correct": correct, "stale": stale}


def summarize(rows):
    by_type = defaultdict(lambda: [0, 0])
    ser = {"current-fact": [0, 0], "current-fact-implicit": [0, 0]}
    for r in rows:
        by_type[r["type"]][1] += 1
        by_type[r["type"]][0] += 1 if r["correct"] else 0
        if r["type"] in ser:
            ser[r["type"]][1] += 1
            ser[r["type"]][0] += 1 if r["stale"] else 0
    acc = {t: round(c / n, 4) if n else 0.0 for t, (c, n) in by_type.items()}
    oc = sum(c for c, n in by_type.values()); on = sum(n for c, n in by_type.values())
    return {
        "overall_acc": round(oc / on, 4) if on else 0.0,
        "acc_by_type": acc,
        "ser_explicit": round(ser["current-fact"][0] / ser["current-fact"][1], 4) if ser["current-fact"][1] else None,
        "ser_implicit": round(ser["current-fact-implicit"][0] / ser["current-fact-implicit"][1], 4) if ser["current-fact-implicit"][1] else None,
        "n": on,
    }


# ----- HoH (external Wikipedia benchmark) support -----------------------------

def load_hoh(n=300, min_outdated=1):
    import datasets
    ds = datasets.load_dataset("russwest404/HoH-QAs", split="train")
    items = []
    for i, rec in enumerate(ds):
        if len(rec.get("outdated_infos") or []) >= min_outdated:
            items.append((i, rec))
        if len(items) >= n:
            break
    return items


def hoh_stream(idx, rec):
    """Ordered evidence docs (oldest outdated -> current last) + the query.
    Each doc's text is the evidence passage as-is (anonymized: no date/outdated
    marker). Mirrors src/hoh_bench.item_to_versions."""
    olds = sorted(rec.get("outdated_infos") or [],
                  key=lambda o: str(o.get("last_modified_time", "")))
    stream = olds + [{"answer": rec["answer"], "evidence": rec["evidence"]}]
    docs = []
    for seq, e in enumerate(stream):
        docs.append({"symbol": e["answer"], "concept": rec["question"],
                     "_text": e.get("evidence") or e["answer"], "ingest_seq": seq})
    query = {"text": rec["question"], "gold": rec["answer"],
             "deprecated_answers": [o["answer"] for o in olds]}
    return docs, query


def _hoh_norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.$%-]", " ", (s or "").lower())).strip()


def _hoh_match(ans, target):
    a, t = _hoh_norm(ans), _hoh_norm(target)
    if not t:
        return False
    if t in a:
        return True
    toks = [w for w in t.split() if len(w) > 2]
    if not toks:
        return t in a
    return sum(1 for w in toks if w in a) / len(toks) >= 0.8


def hoh_score(resp, gold, deprecated):
    correct = _hoh_match(resp, gold)
    stale = (not correct) and any(_hoh_match(resp, d) for d in deprecated)
    return {"correct": correct, "stale": stale}


def hoh_read(context, question, model=None, url=None):
    global _client
    from openai import OpenAI
    model = model or os.environ.get("AGENT_MODEL", "qwen2.5-14b")
    url = url or os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
    if _client is None:
        _client = OpenAI(base_url=url, api_key="dummy")
    system = ("You answer a factual question using ONLY the context. The context may "
              "contain conflicting statements from different times; give the CURRENT "
              "answer. Reply with just the answer, as short as possible.")
    r = _client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=40,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    u = r.usage
    return {"answer": r.choices[0].message.content or "",
            "qtokens": (u.prompt_tokens + u.completion_tokens) if u else 0}


def hoh_ckpt_path(tag):
    return RESULTS / f"_ckpt_{tag}.jsonl"


def hoh_load_done(tag):
    p = hoh_ckpt_path(tag)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def hoh_append(tag, row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(hoh_ckpt_path(tag), "a") as f:
        f.write(json.dumps(row) + "\n")


def hoh_summarize(rows):
    n = len(rows)
    nc = sum(r["correct"] for r in rows); ns = sum(r["stale"] for r in rows)
    return {"n": n, "accuracy": round(nc / n, 4) if n else 0,
            "ser": round(ns / n, 4) if n else 0, "n_correct": nc, "n_stale": ns}


def hoh_write_results(tag, system_name, rows, extra=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    summ = hoh_summarize(rows)
    out = {"tag": tag, "system": system_name, "benchmark": "HoH-QAs",
           "summary": summ, "rows": rows}
    if extra:
        out.update(extra)
    (RESULTS / f"results_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[hoh] {system_name}: acc={summ['accuracy']} SER={summ['ser']} "
          f"(stale={summ['n_stale']}/{summ['n']})")
    return summ


def write_results(tag, system_name, rows, extra=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    summ = summarize(rows)
    out = {"tag": tag, "system": system_name, "summary": summ, "rows": rows}
    if extra:
        out.update(extra)
    path = RESULTS / f"results_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[real] {system_name}: acc={summ['overall_acc']} "
          f"SERx={summ['ser_explicit']} SERi={summ['ser_implicit']} n={summ['n']}")
    print(f"[real] wrote {path}")
    return summ
