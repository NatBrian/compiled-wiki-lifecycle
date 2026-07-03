"""Model-agnostic LLM wrapper. Defaults to a small cached model for CPU smoke tests;
swap MODEL_NAME (or pass model=) for the real GPU run once a card frees.

Non-disruption rule: this never pins a GPU by itself. device='cpu' by default for the
smoke test. For real runs pass device='cuda:<idx>' ONLY for a card confirmed free.
"""
import os

DEFAULT_MODEL = os.environ.get("ABS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


class LLM:
    def __init__(self, model=None, device="cpu", max_new_tokens=256):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model or DEFAULT_MODEL
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        attn = "eager"  # avoid broken cuDNN SDPA kernel on this box
        if device != "cpu":
            try:
                torch.backends.cuda.enable_cudnn_sdp(False)
            except Exception:
                pass
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, attn_implementation=attn
        ).to(device)
        self.model.eval()

    def chat(self, system, user, max_new_tokens=None):
        import torch

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt", truncation=False).to(self.device)
        n_ctx = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][n_ctx:], skip_special_tokens=True)
        return text.strip(), n_ctx

    def chat_batch(self, system, users, max_new_tokens=64, batch_size=64):
        """Batched greedy decode for many same-shape prompts (e.g. compile cards)."""
        import torch

        outs = []
        for s in range(0, len(users), batch_size):
            chunk = users[s:s + batch_size]
            prompts = []
            for u in chunk:
                msgs = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": u}]
                prompts.append(self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            inp = self.tok(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inp, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id)
            gen = out[:, inp["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in outs]

    def context_limit(self):
        return getattr(self.model.config, "max_position_embeddings", 32768)


class OllamaLLM:
    """Backend that routes to a local ollama server (gemma4:12b on GPU 2).

    Same interface as LLM (chat / chat_batch / context_limit) so the drivers don't care.
    Uses /api/chat with think:false (gemma4 emits reasoning tokens otherwise). chat_batch
    exploits OLLAMA_NUM_PARALLEL slots via a small thread pool.
    """

    def __init__(self, model="gemma4:12b", host=None, max_new_tokens=64, ctx=131072,
                 num_parallel=4):
        self.model_name = model
        self.host = host or os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
        self.max_new_tokens = max_new_tokens
        self.ctx = ctx
        self.num_parallel = num_parallel
        self.device = "ollama"

    def _post(self, system, user, max_new_tokens, num_ctx):
        import json, urllib.request
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        payload = {
            "model": self.model_name, "messages": msgs, "stream": False, "think": False,
            "options": {"num_predict": max_new_tokens or self.max_new_tokens,
                        "temperature": 0.0, "num_ctx": num_ctx},
        }
        req = urllib.request.Request(self.host + "/api/chat",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        text = d.get("message", {}).get("content", "")
        n_ctx = d.get("prompt_eval_count", 0)
        return text.strip(), n_ctx

    def chat(self, system, user, max_new_tokens=None, num_ctx=None):
        return self._post(system, user, max_new_tokens, num_ctx or self.ctx)

    def chat_batch(self, system, users, max_new_tokens=64, batch_size=4, num_ctx=4096):
        from concurrent.futures import ThreadPoolExecutor
        out = [None] * len(users)

        def work(i):
            txt, _ = self._post(system, users[i], max_new_tokens, num_ctx)
            return i, txt

        with ThreadPoolExecutor(max_workers=self.num_parallel) as ex:
            for i, txt in ex.map(work, range(len(users))):
                out[i] = txt
        return out

    def context_limit(self):
        return self.ctx
