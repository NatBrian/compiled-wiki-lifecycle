"""Seeded vLLM HTTP client (variance-honesty fix) for the maintain/trained-maintainer stage.

The shared client (code/shared/oai_client.py) does NOT pass a `seed` to the vLLM server,
so at temperature 0.7 the sampling draws are governed by the server's RNG/continuous-batching
state, not by the experiment's --seed (which only shuffles the corpus). Self-critique
correctly flagged that this makes the 3 "seeds" share uncontrolled sampling noise and
understates run-to-run variance.

This wrapper injects a per-run `seed` into every chat request so that (a) a run is
reproducible and (b) DIFFERENT seeds are genuinely independent sampling replications — the
N seeds now capture sampling variance, which is what a variance estimate needs.

We do NOT edit the shared client (other stages depend on its exact behaviour for
reproducibility). We subclass it here, locally to this stage only.
"""
import json
import time
import urllib.error
import urllib.request

from oai_client import VLLM as _BaseVLLM, _HTTPVLLM


class _SeededHTTPVLLM(_HTTPVLLM):
    """_HTTPVLLM that adds an OpenAI-API `seed` to each request for reproducible sampling."""

    def __init__(self, host="http://127.0.0.1:8102", model="qwen2.5-14b", seed=0):
        super().__init__(host=host, model=model)
        self.seed = int(seed)

    def gen(self, system, user, max_new_tokens=600, temperature=0.7, _tries=5):
        body = json.dumps({
            "model": self.model,
            "messages": (([{"role": "system", "content": system}] if system else [])
                         + [{"role": "user", "content": user}]),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
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


def SeededVLLM(host="http://127.0.0.1:8102", model="qwen2.5-14b", seed=0):
    """Factory mirroring oai_client.VLLM. If the offline engine is active, fall back to the
    base client (offline path is deterministic per its own SamplingParams)."""
    import os
    if os.environ.get("CONTRACT_OFFLINE_MODEL"):
        return _BaseVLLM(model=model)
    return _SeededHTTPVLLM(host=host, model=model, seed=seed)
