"""LightRAG (HKUDS/LightRAG, EMNLP 2025) baseline — run the official library as-is.

Only the LLM backend (-> our vLLM Qwen2.5-14B) and embedder (-> local bge-small)
are swapped via LightRAG's documented hooks. We feed it the SAME anonymized
per-release doc stream, take its OWN retrieved context (only_need_context=True),
then score through the shared constant reader. A-priori prediction from source:
ACCUMULATE (descriptions concatenated, no conflict resolution) -> HIGH staleness.

Usage: real/venvs/lightrag/bin/python real/run_lightrag.py <lib>
"""
import os, sys, asyncio, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

LIB = sys.argv[1] if len(sys.argv) > 1 else "pydantic"
MODE = os.environ.get("LIGHTRAG_MODE", "hybrid")
TAG = f"real_lightrag_{LIB}"
WORKDIR = f"/tmp/lightrag_{LIB}"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


async def llm_func(prompt, system_prompt=None, history_messages=None, **kw):
    kw.pop("keyword_extraction", None)
    kw.pop("hashing_kv", None)
    kw.pop("enable_cot", None)
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key="EMPTY", base_url=URL, **kw)


async def emb_func(texts):
    return _emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


async def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    rag = LightRAG(
        working_dir=WORKDIR,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=512, func=emb_func),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except Exception as e:
        print(f"[lightrag] pipeline_status init skipped: {e}")

    versions, bundles, by_version = real_core.load_corpus(LIB)
    rows = []
    for v in versions:
        docs = bundles[v]
        if docs:
            await rag.ainsert([d["_text"] for d in docs])
        for q in by_version.get(v, []):
            if q["type"] == "synthesis":
                continue  # staleness pass; synthesis handled separately (needs judge)
            try:
                ctx = await rag.aquery(q["text"], param=QueryParam(mode=MODE, only_need_context=True))
            except Exception as e:
                ctx = f"(retrieval error: {e})"
            ctx = ctx if isinstance(ctx, str) else str(ctx)
            out = real_core.read_answer(ctx, q["text"], synthesis=False)
            sc = real_core.score_factual(q["gold"], q.get("deprecated_answers", []), out["answer"])
            rows.append({"qid": q["qid"], "type": q["type"], "version": v,
                         "gold": q["gold"], "answer": out["answer"],
                         "correct": sc["correct"], "stale": sc["stale"],
                         "qtokens": out["qtokens"]})
            print(f"  {q['qid']:32} -> {out['answer'][:48]!r} correct={sc['correct']} stale={sc['stale']}")
    real_core.write_results(TAG, f"LightRAG(mode={MODE})", rows,
                            extra={"mode": MODE, "prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    asyncio.run(main())
