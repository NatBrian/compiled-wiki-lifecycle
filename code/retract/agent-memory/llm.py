"""Qwen3-14B client wrapper for the agent-memory stage's experiments.

Talks to the local vLLM OpenAI-compatible server (GPU1, port 8102).
Handles Qwen3 quirks: appends /no_think and strips <think>...</think> blocks.
"""
import os
import re
import time
import openai

BASE_URL = os.environ.get("P7_VLLM_URL", "http://localhost:8102/v1")
MODEL = os.environ.get("P7_MODEL", "Qwen/Qwen3-14B")

_client = openai.OpenAI(base_url=BASE_URL, api_key="x")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks."""
    return _THINK_RE.sub("", text or "").strip()


def chat(messages, temperature=0.0, max_tokens=256, no_think=True, retries=4):
    """Single chat completion. Returns cleaned assistant text.

    messages: list of {role, content} dicts OR a bare user string.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    msgs = [dict(m) for m in messages]
    # /no_think is a Qwen3-only control token; skip for other backbones (e.g. Llama)
    if no_think and "qwen3" in MODEL.lower() and msgs and msgs[-1]["role"] == "user":
        if "/no_think" not in msgs[-1]["content"]:
            msgs[-1]["content"] = msgs[-1]["content"].rstrip() + " /no_think"
    last_err = None
    for attempt in range(retries):
        try:
            r = _client.chat.completions.create(
                model=MODEL, messages=msgs,
                temperature=temperature, max_tokens=max_tokens,
            )
            return strip_think(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")


def ask(question, temperature=0.0, max_tokens=256):
    """Convenience: ask a single question cold (no system prompt, no memory)."""
    return chat([{"role": "user", "content": question}],
                temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    print("smoke:", ask("What is the capital of France?"))
