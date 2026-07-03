"""Mem0 (mem0ai, arXiv 2504.19413) baseline — run the official library as-is.

Mem0 is a CONSOLIDATING memory: on each add() its LLM decides ADD/UPDATE/DELETE/
NOOP against existing memories, so a superseding fact should overwrite/remove the
old one. A-priori prediction: CONSOLIDATE -> LOW staleness. This is the predictive
P4 anchor — a real, stateful, append-capable memory that nonetheless lands on the
GOOD (resolve) side, dissociating "has memory" from "leaks stale".

We feed the SAME anonymized per-release stream (one add() per fact, in version
order), take Mem0's OWN search() results as context, and score through the shared
constant reader.

Usage: real/venvs/mem0/bin/python real/run_mem0.py <lib>
"""
import os, sys, shutil
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

LIB = sys.argv[1] if len(sys.argv) > 1 else "pydantic"
TAG = f"real_mem0_{LIB}"
WORKDIR = f"/tmp/mem0_{LIB}_qdrant"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")
USER = "bench"

from mem0 import Memory

CONFIG = {
    "llm": {"provider": "openai", "config": {
        "model": MODEL, "openai_base_url": URL, "api_key": "dummy", "temperature": 0.0}},
    "embedder": {"provider": "huggingface", "config": {
        "model": "BAAI/bge-small-en-v1.5"}},
    "vector_store": {"provider": "qdrant", "config": {
        "path": WORKDIR, "collection_name": "bench", "embedding_model_dims": 384}},
}


def get_memories(res):
    if isinstance(res, dict):
        res = res.get("results", [])
    out = []
    for m in res:
        if isinstance(m, dict):
            out.append(m.get("memory") or m.get("text") or str(m))
        else:
            out.append(str(m))
    return out


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    mem = Memory.from_config(CONFIG)
    versions, bundles, by_version = real_core.load_corpus(LIB)
    rows = []
    for v in versions:
        for d in bundles[v]:
            try:
                mem.add([{"role": "user", "content": d["_text"]}], user_id=USER)
            except Exception as e:
                print(f"  add fail {d['id']}: {e}")
        for q in by_version.get(v, []):
            if q["type"] == "synthesis":
                continue
            try:
                res = mem.search(q["text"], user_id=USER, limit=5)
                ctx = "\n".join(get_memories(res)) or "(no memories)"
            except Exception as e:
                ctx = f"(search error: {e})"
            out = real_core.read_answer(ctx, q["text"], synthesis=False)
            sc = real_core.score_factual(q["gold"], q.get("deprecated_answers", []), out["answer"])
            rows.append({"qid": q["qid"], "type": q["type"], "version": v,
                         "gold": q["gold"], "answer": out["answer"],
                         "correct": sc["correct"], "stale": sc["stale"],
                         "qtokens": out["qtokens"]})
            print(f"  {q['qid']:32} -> {out['answer'][:48]!r} correct={sc['correct']} stale={sc['stale']}")
    real_core.write_results(TAG, "Mem0", rows, extra={"prediction": "consolidate->LOW"})


if __name__ == "__main__":
    main()
