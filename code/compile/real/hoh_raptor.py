"""RAPTOR on HoH (n=300), fresh tree per item, checkpoint+resume.
Run (cwd=benchmark dir): real/venvs/raptor/bin/python real/hoh_raptor.py [N]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "raptor_src"))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = "real_raptor_hoh"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from raptor import (RetrievalAugmentation, RetrievalAugmentationConfig,
                    BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel)
_client = OpenAI(base_url=URL, api_key="dummy")
_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


class VS(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        r = _client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": f"Summarize concisely:\n{context}"}])
        return r.choices[0].message.content or ""

class VQ(BaseQAModel):
    def answer_question(self, context, question):
        return ""   # unused; constant reader scores

class EB(BaseEmbeddingModel):
    def create_embedding(self, text):
        return _emb.encode(text)


def context_for(item_docs, question):
    cfg = RetrievalAugmentationConfig(summarization_model=VS(), qa_model=VQ(), embedding_model=EB())
    RA = RetrievalAugmentation(config=cfg)
    RA.add_documents("\n".join(d["_text"] for d in item_docs))
    try:
        out = RA.retrieve(question)
        return out[0] if isinstance(out, tuple) else out
    except Exception as e:
        return f"(retrieval error: {e})"


def main():
    items = real_core.load_hoh(N)
    done = real_core.hoh_load_done(TAG)
    rows = list(done); start = len(done)
    print(f"[raptor-hoh] {len(items)} items, resuming from {start}", flush=True)
    for idx in range(start, len(items)):
        i, rec = items[idx]
        docs, query = real_core.hoh_stream(i, rec)
        ctx = context_for(docs, query["text"])
        out = real_core.hoh_read(str(ctx), query["text"])
        sc = real_core.hoh_score(out["answer"], query["gold"], query["deprecated_answers"])
        row = {"q": rec["question"][:80], "gold": query["gold"], "resp": out["answer"][:80],
               **sc, "n_dep": len(query["deprecated_answers"])}
        rows.append(row); real_core.hoh_append(TAG, row)
        if (idx + 1) % 25 == 0:
            s = real_core.hoh_summarize(rows)
            print(f"  [{idx+1}/{len(items)}] acc={s['accuracy']} SER={s['ser']}", flush=True)
    real_core.hoh_write_results(TAG, "RAPTOR", rows, extra={"prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    main()
