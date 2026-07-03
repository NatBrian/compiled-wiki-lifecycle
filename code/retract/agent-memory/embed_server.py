"""Tiny OpenAI-compatible /v1/embeddings shim (CPU bge-small) for Letta.

Our vLLM serves Qwen3-14B (generation only). Letta needs an embeddings endpoint
for its recall/archival memory. This serves BAAI/bge-small-en-v1.5 on CPU in the
OpenAI embeddings wire format so Letta can use embedding_endpoint_type="openai"
pointed here -- no extra GPU.
Run: python code/embed_server.py  (port 8290)
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import uvicorn  # noqa: E402

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
_model = SentenceTransformer(MODEL_NAME, device="cpu")
app = FastAPI()


class EmbReq(BaseModel):
    input: object
    model: str = MODEL_NAME


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbReq):
    inp = req.input
    texts = [inp] if isinstance(inp, str) else list(inp)
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    vecs = _model.encode(texts, normalize_embeddings=True)
    data = [{"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": MODEL_NAME,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("P7_EMB_PORT", "8290")),
                log_level="warning")
