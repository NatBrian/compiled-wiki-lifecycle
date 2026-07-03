"""Mem0 on HoH (n=300), fresh memory per item, checkpoint+resume.
Run: real/venvs/mem0/bin/python real/hoh_mem0.py [N]"""
import os, sys, shutil
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = "real_mem0_hoh"
WORK = "/tmp/mem0_hoh_item_qdrant"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")
USER = "hoh"

from mem0 import Memory
CONFIG = {
    "llm": {"provider": "openai", "config": {"model": MODEL, "openai_base_url": URL,
                                             "api_key": "dummy", "temperature": 0.0}},
    "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}},
    "vector_store": {"provider": "qdrant", "config": {
        "path": WORK, "collection_name": "hoh", "embedding_model_dims": 384}},
}


def get_memories(res):
    res = res.get("results", res) if isinstance(res, dict) else res
    out = []
    for m in res:
        out.append((m.get("memory") or m.get("text") or str(m)) if isinstance(m, dict) else str(m))
    return out


def context_for(item_docs, question):
    shutil.rmtree(WORK, ignore_errors=True)
    mem = Memory.from_config(CONFIG)
    for d in item_docs:
        try:
            mem.add([{"role": "user", "content": d["_text"]}], user_id=USER)
        except Exception as e:
            print(f"  add fail: {e}")
    try:
        ctx = "\n".join(get_memories(mem.search(question, user_id=USER, limit=5))) or "(no memories)"
    except Exception as e:
        ctx = f"(search error: {e})"
    return ctx


def main():
    items = real_core.load_hoh(N)
    done = real_core.hoh_load_done(TAG)
    rows = list(done); start = len(done)
    print(f"[mem0-hoh] {len(items)} items, resuming from {start}", flush=True)
    for idx in range(start, len(items)):
        i, rec = items[idx]
        docs, query = real_core.hoh_stream(i, rec)
        ctx = context_for(docs, query["text"])
        out = real_core.hoh_read(ctx, query["text"])
        sc = real_core.hoh_score(out["answer"], query["gold"], query["deprecated_answers"])
        row = {"q": rec["question"][:80], "gold": query["gold"], "resp": out["answer"][:80],
               **sc, "n_dep": len(query["deprecated_answers"])}
        rows.append(row); real_core.hoh_append(TAG, row)
        if (idx + 1) % 25 == 0:
            s = real_core.hoh_summarize(rows)
            print(f"  [{idx+1}/{len(items)}] acc={s['accuracy']} SER={s['ser']}", flush=True)
    real_core.hoh_write_results(TAG, "Mem0", rows, extra={"prediction": "consolidate-but-recency-blind"})


if __name__ == "__main__":
    main()
