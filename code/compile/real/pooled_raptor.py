"""Pooled RAPTOR (parthsarthi03/raptor, ICLR 2024 — official repo as-is) on the
shared HoH corpus: the second REAL accumulate-structured system (de-risks C2,
"not structure").

ALL 628 evidence docs go into ONE RAPTOR tree in global ingest order (single
timeline). RAPTOR clusters + summarizes hierarchically and has no supersession
mechanism, so current and stale leaves coexist and summaries blend them.
A-priori prediction (registered in DESIGN.md taxonomy): ACCUMULATE -> HIGH SER.

Run: cd benchmark-dynamic-corpus && real/venvs/raptor/bin/python real/pooled_raptor.py [N]
Tree is saved to /tmp/raptor_pooled_tree after build (resume-safe for queries).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "raptor_src"))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = "pool_raptor"
TREE = "/tmp/raptor_pooled_tree"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from raptor import (RetrievalAugmentation, RetrievalAugmentationConfig,
                    BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel,
                    TreeRetriever)
_client = OpenAI(base_url=URL, api_key="dummy")
_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


class VS(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        r = _client.chat.completions.create(model=MODEL, temperature=0.0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": f"Summarize concisely:\n{context}"}])
        return r.choices[0].message.content or ""


class VQ(BaseQAModel):
    def answer_question(self, context, question):
        return ""  # unused; the shared constant reader scores


class EB(BaseEmbeddingModel):
    def create_embedding(self, text):
        return _emb.encode(text)


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


def main():
    docs, queries = build_pool()
    print(f"[pool-raptor] {len(docs)} docs, {len(queries)} queries", flush=True)
    cfg = RetrievalAugmentationConfig(summarization_model=VS(), qa_model=VQ(),
                                      embedding_model=EB())
    if os.path.exists(TREE):
        RA = RetrievalAugmentation(config=cfg, tree=TREE)
        print("[pool-raptor] tree loaded from disk", flush=True)
    else:
        RA = RetrievalAugmentation(config=cfg)
        blob = "\n\n".join(d["_text"] for d in sorted(docs, key=lambda x: x["ingest_seq"]))
        # single-threaded build: server under heavy load, multithreaded leaf
        # creation hits EAGAIN on thread spawn (rayon/tokenizers panic)
        RA.tree = RA.tree_builder.build_from_text(text=blob, use_multithreading=False)
        RA.retriever = TreeRetriever(RA.tree_retriever_config, RA.tree)
        try:
            RA.save(TREE)
            print("[pool-raptor] tree built + saved", flush=True)
        except Exception as e:
            print(f"[pool-raptor] tree built (save failed: {e})", flush=True)

    rows = real_core.hoh_load_done(TAG)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        try:
            ctx = RA.retrieve(q["text"])
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
    real_core.hoh_write_results(TAG, "pooled-RAPTOR", rows,
                                extra={"prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    main()
