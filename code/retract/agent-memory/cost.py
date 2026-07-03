"""Per-hook cost/latency of the gate variants -- the 'weight-free tax' (Table 8).

Measures wall-clock + token cost of one membership-check at the write/consolidate
hook for each predicate:
  hash : normalized substring test          (no model call)
  cone : 1 embedding + cosine               (CPU embedder)
  nli  : 1 LLM entailment call              (vLLM)
Reports mean/median latency over R trials on representative memory notes.
Output: results/cost.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import llm
import mem0_backend as mb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = int(os.environ.get("P7_COST_R", "30"))

NOTE = "Morzellelle Pliquistish's current account balance is $9,596,288"
VALUE = "$9,596,288"
Q = "What is the account balance?"


def t_hash():
    t = time.perf_counter()
    mb.contains_value(NOTE, VALUE)
    return time.perf_counter() - t


def t_cone(embedder):
    t = time.perf_counter()
    a = embedder.embed(NOTE); b = embedder.embed(VALUE)
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5; db = sum(y * y for y in b) ** 0.5
    _ = num / (da * db + 1e-9)
    return time.perf_counter() - t


def t_nli():
    t = time.perf_counter()
    llm.ask(f"Does this note reveal the answer to '{Q}'? Note: {NOTE}\nAnswer YES or NO.",
            temperature=0.0, max_tokens=4)
    return time.perf_counter() - t


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"mean_ms": round(1000 * sum(xs) / n, 3),
            "median_ms": round(1000 * xs[n // 2], 3),
            "p90_ms": round(1000 * xs[min(n - 1, int(0.9 * n))], 3), "n": n}


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mem0.embeddings.huggingface import HuggingFaceEmbedding
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    emb = HuggingFaceEmbedding(BaseEmbedderConfig(model="BAAI/bge-small-en-v1.5"))
    # warmup
    t_cone(emb); t_nli()
    out = {"R": R,
           "hash": stats([t_hash() for _ in range(R)]),
           "cone": stats([t_cone(emb) for _ in range(R)]),
           "nli": stats([t_nli() for _ in range(R)])}
    out["note"] = ("weight-free tax per write-hook: hash ~free, cone = 1 CPU embed, "
                   "nli = 1 vLLM call; A3-=hash+cone is the cheap default, full A3 adds nli")
    json.dump(out, open(f"{ROOT}/results/cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    os._exit(0)
