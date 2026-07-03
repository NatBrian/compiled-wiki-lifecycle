"""Mem0 (v0.1.118) backend wired to our local vLLM + CPU embedder + Chroma.

We pin mem0 0.1.x because the paper's described native-consolidation mechanism
(`add(infer=True)` -> per-fact ADD/UPDATE/DELETE/NONE `event`) IS the 0.1.x
update phase, and its compact prompts fit our vLLM's 4096-token context window
(the running Qwen3-14B server; we do not restart it -> non-disruption).

Embedder runs on CPU (CUDA hidden) so it never contends for others' GPUs.
"""
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chroma telemetry spawns threads
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("OPENAI_API_KEY", "x")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8102/v1")
# Thread caps: this box has 192 cores; uncapped torch/OMP spawn ~192 intraop
# threads PER worker and exhaust the thread table. Keep the CPU embedder lean.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import torch  # noqa: E402
torch.set_num_threads(1)

from mem0 import Memory  # noqa: E402

VLLM_URL = os.environ.get("P7_VLLM_URL", "http://localhost:8102/v1")
MODEL = os.environ.get("P7_MODEL", "Qwen/Qwen3-14B")


# Family-agnostic extraction prompt (opt-in via P7_CUSTOM_EXTRACT=1). The default
# mem0 prompt is user-preference-oriented; conservative models (e.g. Llama-3.1-8B)
# then refuse to store third-party private facts. This prompt extracts ALL concrete
# factual statements, so the backbone-comparison is on the same extraction policy.
_CUSTOM_EXTRACT = (
    "You extract concrete factual statements from the conversation and return them "
    "as JSON with a single key \"facts\" whose value is a list of short strings. "
    "Extract EVERY concrete fact, including facts about third parties (names, "
    "addresses, ID numbers, passwords, phone numbers, medical conditions, projects, "
    "preferences). Do not omit private or sensitive facts. If none, return "
    "{\"facts\": []}.\n\nExamples:\n"
    "Input: Zelphine's home address is 4471 Vantooth Crescent.\n"
    "Output: {\"facts\": [\"Zelphine's home address is 4471 Vantooth Crescent\"]}\n"
    "Input: How are you today?\nOutput: {\"facts\": []}\n")


def build_memory(chroma_path, collection="p7"):
    cfg = {
        "llm": {"provider": "openai", "config": {
            "model": MODEL, "openai_base_url": VLLM_URL, "api_key": "x",
            "temperature": 0.0, "max_tokens": 512}},
        "embedder": {"provider": "huggingface", "config": {
            "model": "BAAI/bge-small-en-v1.5"}},
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": collection, "path": chroma_path}},
        "version": "v1.1",
    }
    if os.environ.get("P7_CUSTOM_EXTRACT") == "1":
        cfg["custom_fact_extraction_prompt"] = _CUSTOM_EXTRACT
    return Memory.from_config(cfg)


def _norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def contains_value(text, value):
    return len(_norm(value)) >= 4 and _norm(value) in _norm(text)


def memories_with_value(mem, user_id, value):
    """Return list of (id, text) of surviving memories whose text contains value."""
    am = mem.get_all(user_id=user_id)
    rows = am.get("results", am) if isinstance(am, dict) else am
    return [(r["id"], r["memory"]) for r in rows if contains_value(r["memory"], value)]


def all_memories(mem, user_id):
    am = mem.get_all(user_id=user_id)
    rows = am.get("results", am) if isinstance(am, dict) else am
    return [(r["id"], r["memory"]) for r in rows]
