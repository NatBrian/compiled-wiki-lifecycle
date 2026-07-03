"""Graphiti (getzep/graphiti, Zep, arXiv 2501.13956) baseline — official lib as-is.

Graphiti is a bi-temporal knowledge-graph memory: when a new fact contradicts an
existing one it INVALIDATES the old edge (sets invalid_at) rather than dropping it,
and search returns only currently-valid edges. Crucially it is RECENCY-AWARE (it
uses reference_time), unlike Mem0. A-priori prediction: temporal invalidation ->
LOW staleness. This is the decisive complement to Mem0 (real consolidating memory
that DOES carry recency -> should land on the resolve/LOW side).

Runs fully embedded: KuzuDriver(':memory:') (no Neo4j/Java), local bge-small
embedder, our vLLM as the LLM. We feed the SAME anon stream as ordered episodes
(reference_time increasing with version), take Graphiti's OWN valid-edge search as
context, and score through the shared constant reader.

Run: real/venvs/graphiti/bin/python real/run_graphiti.py <lib>
"""
import os, sys, asyncio
os.environ.setdefault("EMBEDDING_DIM", "384")       # bge-small; Kuzu uses FLOAT[] (any dim)
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core
from datetime import datetime, timedelta, timezone
import numpy as np

LIB = sys.argv[1] if len(sys.argv) > 1 else "pydantic"
TAG = f"real_graphiti_{LIB}"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")
BASE_T = datetime(2020, 1, 1, tzinfo=timezone.utc)

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.search.search_config import (SearchConfig, EdgeSearchConfig,
                                                EdgeSearchMethod, EdgeReranker)
from sentence_transformers import SentenceTransformer

# vector-only edge search: Kuzu BM25/full-text needs an FTS extension download
# (network-blocked here), so we use cosine_similarity only. Currency still comes
# from Graphiti's temporal edge invalidation, which is what we are testing.
VECTOR_ONLY = SearchConfig(
    edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.cosine_similarity],
                                 reranker=EdgeReranker.rrf),
    limit=5)

_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


class SBEmbedder(EmbedderClient):
    def __init__(self):
        self.config = EmbedderConfig(embedding_dim=384)

    async def create(self, input_data):
        if isinstance(input_data, str):
            return _emb.encode(input_data, normalize_embeddings=True).tolist()
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            return _emb.encode(input_data, normalize_embeddings=True).tolist()
        v = _emb.encode(str(input_data), normalize_embeddings=True)
        return v.tolist()

    async def create_batch(self, input_data_list):
        return _emb.encode(input_data_list, normalize_embeddings=True).tolist()


class SBCrossEncoder(CrossEncoderClient):
    """Local embedding-cosine reranker (avoids a remote reranker API)."""
    async def rank(self, query, passages):
        if not passages:
            return []
        q = _emb.encode(query, normalize_embeddings=True)
        ps = _emb.encode(passages, normalize_embeddings=True)
        scores = (ps @ q).tolist()
        return sorted(zip(passages, scores), key=lambda t: t[1], reverse=True)


def make_client():
    # cap output tokens: vLLM ctx=16384 total; Graphiti's default 16384 OUTPUT leaves no room for input
    llm = OpenAIGenericClient(config=LLMConfig(
        api_key="dummy", model=MODEL, small_model=MODEL, base_url=URL, max_tokens=2048),
        max_tokens=2048)
    return Graphiti(graph_driver=KuzuDriver(db=":memory:"),
                    llm_client=llm, embedder=SBEmbedder(), cross_encoder=SBCrossEncoder())


async def main():
    g = make_client()
    await g.build_indices_and_constraints()
    versions, bundles, by_version = real_core.load_corpus(LIB)
    rows = []
    for v in versions:
        for d in bundles[v]:
            # SAME anon content as every other system, phrased as a statement so
            # Graphiti's relation extractor yields an edge (the unit its temporal
            # invalidation acts on). No version/supersession info added.
            body = f"To {d['concept']}, use {d['symbol']}."
            try:
                await g.add_episode(
                    name=d["id"], episode_body=body,
                    source=EpisodeType.text, source_description=f"release v{v}",
                    reference_time=BASE_T + timedelta(days=d["ingest_seq"]))
            except Exception as e:
                print(f"  add_episode fail {d['id']}: {e}")
        for q in by_version.get(v, []):
            if q["type"] == "synthesis":
                continue
            try:
                res = await g.search_(q["text"], config=VECTOR_ONLY)
                edges = res.edges
                ctx = "\n".join(getattr(e, "fact", str(e)) for e in edges) or "(no facts)"
            except Exception as e:
                ctx = f"(search error: {e})"
            out = real_core.read_answer(ctx, q["text"], synthesis=False)
            sc = real_core.score_factual(q["gold"], q.get("deprecated_answers", []), out["answer"])
            rows.append({"qid": q["qid"], "type": q["type"], "version": v,
                         "gold": q["gold"], "answer": out["answer"],
                         "correct": sc["correct"], "stale": sc["stale"], "qtokens": out["qtokens"]})
            print(f"  {q['qid']:32} -> {out['answer'][:42]!r} correct={sc['correct']} stale={sc['stale']}")
    real_core.write_results(TAG, "Graphiti", rows, extra={"prediction": "temporal-invalidation->LOW"})


if __name__ == "__main__":
    asyncio.run(main())
