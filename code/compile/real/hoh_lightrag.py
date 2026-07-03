"""LightRAG on HoH (n=300), fresh instance per item, checkpoint+resume.
Run: real/venvs/lightrag/bin/python real/hoh_lightrag.py [N]"""
import os, sys, asyncio, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = "real_lightrag_hoh"
WORK = "/tmp/lightrag_hoh_item"
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


async def context_for(item_docs, question):
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK, exist_ok=True)
    rag = LightRAG(working_dir=WORK, llm_model_func=llm_func,
                   embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=512, func=emb_func))
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except Exception:
        pass
    await rag.ainsert([d["_text"] for d in item_docs])
    try:
        ctx = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_context=True))
    except Exception as e:
        ctx = f"(retrieval error: {e})"
    return ctx if isinstance(ctx, str) else str(ctx)


async def main():
    items = real_core.load_hoh(N)
    done = real_core.hoh_load_done(TAG)
    rows = list(done)
    start = len(done)
    print(f"[lightrag-hoh] {len(items)} items, resuming from {start}", flush=True)
    for idx in range(start, len(items)):
        i, rec = items[idx]
        docs, query = real_core.hoh_stream(i, rec)
        ctx = await context_for(docs, query["text"])
        out = real_core.hoh_read(ctx, query["text"])
        sc = real_core.hoh_score(out["answer"], query["gold"], query["deprecated_answers"])
        row = {"q": rec["question"][:80], "gold": query["gold"], "resp": out["answer"][:80],
               **sc, "n_dep": len(query["deprecated_answers"])}
        rows.append(row); real_core.hoh_append(TAG, row)
        if (idx + 1) % 25 == 0:
            s = real_core.hoh_summarize(rows)
            print(f"  [{idx+1}/{len(items)}] acc={s['accuracy']} SER={s['ser']}", flush=True)
    real_core.hoh_write_results(TAG, "LightRAG", rows, extra={"prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    asyncio.run(main())
