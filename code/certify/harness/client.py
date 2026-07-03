"""HTTP client for the already-resident llama-server (ollama-spawned) on GPU2.

Non-disruption: we do NOT launch a new GPU process. We query an endpoint that is
already loaded (pid owned by us). All inference is light, bursty HTTP. No new
GPU footprint, no contention beyond what already exists.

The server speaks the OpenAI /v1/chat/completions schema. -np 4 => up to 4
concurrent slots, so we fan out with a small thread pool.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Stable ollama endpoint (:11434) survives model reloads / ephemeral llama-server
# port changes. Do NOT point at the transient llama-server port.
HOST = os.environ.get("ABS_LLM_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("ABS_LLM_MODEL", "gemma4:12b")
# harmony/think models emit a hidden 'thought' channel before the answer; strip it
_THINK = re.compile(r"<\|?channel\|?>.*?(?:<\|?channel\|?>|$)", re.DOTALL)


class HTTPLLM:
    def __init__(self, host=HOST, model=MODEL, max_workers=4):
        self.host = host.rstrip("/")
        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.n_calls = 0
        self._warm()

    def _warm(self):
        """Pin the model in VRAM for 30m so ollama does not idle-unload mid-run
        (an unload triggers a reload blip + ephemeral port change)."""
        body = json.dumps({"model": self.model, "keep_alive": "30m",
                           "messages": [{"role": "user", "content": "ok"}],
                           "max_tokens": 1}).encode()
        try:
            req = urllib.request.Request(self.host + "/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=180).read()
        except Exception:
            pass

    def _one(self, system, user, max_new_tokens, _tries=4):
        # native /api/chat with think=False -> no reasoning tokens, fast clean answer
        body = json.dumps({
            "model": self.model,
            "messages": (([{"role": "system", "content": system}] if system else [])
                         + [{"role": "user", "content": user}]),
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": max_new_tokens},
        }).encode()
        last = None
        for attempt in range(_tries):
            try:
                req = urllib.request.Request(
                    self.host + "/api/chat", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                txt = d.get("message", {}).get("content", "")
                return _THINK.sub("", txt).strip()
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last = e  # ollama idle-reload blip -> retry w/ backoff
                time.sleep(2 * (attempt + 1))
        raise last

    def chat(self, system, user, max_new_tokens=64):
        self.n_calls += 1
        return self._one(system, user, max_new_tokens), self._approx_tokens(system, user)

    def chat_batch(self, system, users, max_new_tokens=64, batch_size=4):
        """Concurrent fan-out over the server's parallel slots. Order preserved."""
        self.n_calls += len(users)
        return list(self.pool.map(lambda u: self._one(system, u, max_new_tokens), users))

    @staticmethod
    def _approx_tokens(system, user):
        return ((len(system) if system else 0) + len(user)) // 4

    def context_limit(self):
        return 16384
