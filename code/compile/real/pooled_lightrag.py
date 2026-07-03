"""Pooled LightRAG (the 6th, real published framework) on the shared HoH corpus.

ALL 628 evidence docs inserted into ONE LightRAG instance in global ingest order
(single timeline) -> one graph build -> 300 queries against it. LightRAG ACCUMULATES
entities/relations, so stale and current facts coexist in the graph; prediction is
high staleness (accumulate -> HIGH), matching the per-item LightRAG result and
contrasting the curation arms (wiki/resolve) on the SAME pooled corpus.

Run: real/venvs/lightrag/bin/python real/pooled_lightrag.py [N]
"""
import os, sys, asyncio, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = "pool_lightrag"
WORK = "/tmp/lightrag_pooled"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


async def llm_func(prompt, system_prompt=None, history_messages=None, **kw):
    for k in ("keyword_extraction", "hashing_kv", "enable_cot"):
        kw.pop(k, None)
    return await openai_complete_if_cache(MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages or [], api_key="EMPTY", base_url=URL, **kw)

async def emb_func(texts):
    return _emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def build_pool():
    items = real_core.load_hoh(N)
    docs, queries = [], []
    gseq = 0
    for cid, (i, rec) in enumerate(items):
        stream, query = real_core.hoh_stream(i, rec)
        for d in stream:
            d = dict(d); d["ingest_seq"] = gseq; gseq += 1
            docs.append(d)
        queries.append(query)
    return docs, queries


async def main():
    docs, queries = build_pool()
    print(f"[pool-lightrag] {len(docs)} docs, {len(queries)} queries", flush=True)
    # one graph over the whole pool. RESUME-SAFE: never wipe an existing WORK dir
    # (lightrag.ainsert is idempotent — it tracks doc status and skips already-
    # processed docs, and its LLM cache makes re-extraction cache-fast). The marker
    # only signals "full insert finished" so we can skip ainsert entirely.
    built_marker = os.path.join(WORK, ".pool_built")
    os.makedirs(WORK, exist_ok=True)
    rag = LightRAG(working_dir=WORK, llm_model_func=llm_func,
                   embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=512, func=emb_func))
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except Exception:
        pass
    if not os.path.exists(built_marker):
        # insert in ingest order so the graph's newest mentions are the current facts
        await rag.ainsert([d["_text"] for d in sorted(docs, key=lambda x: x["ingest_seq"])])
        open(built_marker, "w").write("ok")
        print("[pool-lightrag] graph built", flush=True)

    rows = real_core.hoh_load_done(TAG)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        try:
            ctx = await rag.aquery(q["text"], param=QueryParam(mode="hybrid", only_need_context=True))
        except Exception as e:
            ctx = f"(retrieval error: {e})"
        ctx = ctx if isinstance(ctx, str) else str(ctx)
        out = real_core.hoh_read(ctx, q["text"])
        sc = real_core.hoh_score(out["answer"], q["gold"], q["deprecated_answers"])
        row = {"gold": q["gold"], "resp": out["answer"][:80], **sc,
               "n_dep": len(q["deprecated_answers"])}
        rows.append(row); real_core.hoh_append(TAG, row)
        if len(rows) % 25 == 0:
            s = real_core.hoh_summarize(rows)
            print(f"  [{len(rows)}/{len(queries)}] acc={s['accuracy']} SER={s['ser']}", flush=True)
    real_core.hoh_write_results(TAG, "pooled-LightRAG", rows, extra={"prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    asyncio.run(main())
