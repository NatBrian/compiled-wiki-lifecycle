"""RAPTOR (parthsarthi03/raptor, ICLR 2024) baseline — official repo as-is.

RAPTOR builds a hierarchical cluster-and-summarize tree. It has no incremental
update, so at each release we rebuild the tree from the cumulative anonymized
corpus up to that version (which is exactly the accumulate setting: old + new
leaves coexist, summaries are built over both). We take RAPTOR's OWN retrieval
(RA.retrieve) as context and score through the shared constant reader.
A-priori prediction from design: ACCUMULATE -> HIGH staleness.

Run: real/venvs/raptor/bin/python real/run_raptor.py <lib>   (cwd = real/raptor_src)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # real_core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "raptor_src"))
import real_core

LIB = sys.argv[1] if len(sys.argv) > 1 else "pydantic"
TAG = f"real_raptor_{LIB}"
URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from raptor import (RetrievalAugmentation, RetrievalAugmentationConfig,
                    BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel)

_client = OpenAI(base_url=URL, api_key="dummy")
_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


class VLLMSummarizer(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        r = _client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": f"Summarize concisely:\n{context}"}])
        return r.choices[0].message.content or ""


class VLLMQA(BaseQAModel):                       # required by config; unused (we use constant reader)
    def answer_question(self, context, question):
        r = _client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=64,
            messages=[{"role": "user", "content": f"Context:{context}\nQ:{question}"}])
        return r.choices[0].message.content or ""


class SBEmbed(BaseEmbeddingModel):
    def create_embedding(self, text):
        return _emb.encode(text)


def build_tree(texts):
    cfg = RetrievalAugmentationConfig(
        summarization_model=VLLMSummarizer(), qa_model=VLLMQA(), embedding_model=SBEmbed())
    RA = RetrievalAugmentation(config=cfg)
    RA.add_documents("\n".join(texts))
    return RA


def get_context(RA, question):
    try:
        out = RA.retrieve(question)
        return out[0] if isinstance(out, tuple) else out
    except Exception as e:
        return f"(retrieval error: {e})"


def main():
    versions, bundles, by_version = real_core.load_corpus(LIB)
    rows, cum = [], []
    for v in versions:
        cum += [d["_text"] for d in bundles[v]]
        qs = [q for q in by_version.get(v, []) if q["type"] != "synthesis"]
        if not qs:
            continue
        RA = build_tree(cum)            # rebuild on cumulative corpus (RAPTOR has no incremental)
        for q in qs:
            ctx = get_context(RA, q["text"])
            out = real_core.read_answer(str(ctx), q["text"], synthesis=False)
            sc = real_core.score_factual(q["gold"], q.get("deprecated_answers", []), out["answer"])
            rows.append({"qid": q["qid"], "type": q["type"], "version": v,
                         "gold": q["gold"], "answer": out["answer"],
                         "correct": sc["correct"], "stale": sc["stale"], "qtokens": out["qtokens"]})
            print(f"  {q['qid']:32} -> {out['answer'][:48]!r} correct={sc['correct']} stale={sc['stale']}")
    real_core.write_results(TAG, "RAPTOR", rows, extra={"prediction": "accumulate->HIGH"})


if __name__ == "__main__":
    main()
