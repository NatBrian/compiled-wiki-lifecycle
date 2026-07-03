"""Chat client for the contract experiments.

Two backends:
  - HTTP (default): talks to a local vLLM server (/v1/chat/completions).
  - OFFLINE (env CONTRACT_OFFLINE_MODEL set): loads the model IN-PROCESS via vLLM's
    offline LLM engine. Needed when the environment forbids long-running background
    GPU servers -- each experiment runs as one foreground process that loads the
    model once. Concurrent .gen() calls are serialized through one engine (a lock),
    so callers should pass workers=1.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from vllm import LLM
        model = os.environ["CONTRACT_OFFLINE_MODEL"]
        _ENGINE = LLM(model=model, gpu_memory_utilization=float(os.environ.get("CONTRACT_GPU_UTIL", "0.55")),
                      max_model_len=int(os.environ.get("CONTRACT_MAXLEN", "12288")),
                      enforce_eager=True, disable_log_stats=True)
    return _ENGINE


class OfflineVLLM:
    """In-process vLLM offline engine with the same .gen() interface."""

    def __init__(self, model=None):
        self.model = model or os.environ.get("CONTRACT_OFFLINE_MODEL", "offline")
        self.n_calls = 0
        self._eng = _get_engine()

    def gen(self, system, user, max_new_tokens=600, temperature=0.7, _tries=5):
        return self.gen_batch([(system, user)], max_new_tokens, temperature)[0]

    def gen_batch(self, items, max_new_tokens=600, temperature=0.7):
        """Batched generation: items = list of (system, user). One engine pass -> fast."""
        from vllm import SamplingParams
        convs = [(([{"role": "system", "content": s}] if s else []) +
                  [{"role": "user", "content": u}]) for s, u in items]
        sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=0)
        with _ENGINE_LOCK:
            outs = self._eng.chat(convs, sp, use_tqdm=False)
        self.n_calls += len(items)
        return [o.outputs[0].text.strip() for o in outs]


def VLLM(host="http://127.0.0.1:8102", model="qwen2.5-14b"):
    """Factory: offline in-process engine if CONTRACT_OFFLINE_MODEL is set, else HTTP."""
    if os.environ.get("CONTRACT_OFFLINE_MODEL"):
        return OfflineVLLM(model=model)
    return _HTTPVLLM(host=host, model=model)


class _HTTPVLLM:
    def __init__(self, host="http://127.0.0.1:8102", model="qwen2.5-14b"):
        self.host = host
        self.model = model
        self.n_calls = 0

    def gen(self, system, user, max_new_tokens=600, temperature=0.7, _tries=5):
        body = json.dumps({
            "model": self.model,
            "messages": (([{"role": "system", "content": system}] if system else [])
                         + [{"role": "user", "content": user}]),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }).encode()
        last = None
        for attempt in range(_tries):
            try:
                req = urllib.request.Request(
                    self.host + "/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.loads(r.read())
                self.n_calls += 1
                return d["choices"][0]["message"]["content"].strip()
            except (urllib.error.URLError, ConnectionError, OSError, KeyError) as e:
                last = e
                time.sleep(3 * (attempt + 1))
        raise last
