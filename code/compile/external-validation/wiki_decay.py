"""Post 1 experiment — "AI that rewrites its own knowledge quietly destroys it."

Reproduces, on a REAL public LLM-Wiki workflow + REAL data, the core finding:
iterated rewriting of a compiled knowledge store silently loses facts, a bigger model
doesn't fix it, and a cheap check-and-rollback harness does.

- Tool: the rewrite loop + prompts are faithful to the Hermes LLM-Wiki skill
  (NousResearch/hermes-agent, skills/research/llm-wiki) and Karpathy's LLM-Wiki pattern:
  one page per concept, "update facts; newer supersedes" on each new source.
- Data: SciFact-Open (Wadden et al.) — real PubMed abstracts + expert claims.
  Subset prebuilt by build_subset.py -> data/scifact_subset.json (self-contained for Kaggle).
- Metric: retention(t) = fraction of tracked claims still SUPPORTED by the wiki after t
  rewrite rounds, judged by the same model (temperature 0). Loss = 1 - retention.

Runs on a single 16GB GPU (Kaggle T4) with a 4-bit Qwen3 model. Checkpoints every round.
"""
import argparse, json, os, time, random

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Prompts: the wiki workflow is taken from the REAL Hermes llm-wiki skill, not invented. ----
# Source: NousResearch/hermes-agent  skills/research/llm-wiki/SKILL.md  @ commit d1383a6 (2026-06-10).
# Karpathy's LLM-Wiki gist (442a6bf): the LLM "reads it, extracts the key information, and integrates
#   it into the existing wiki." Hermes ingest step: "Add new information, update facts, bump updated
#   date." Hermes Update Policy (verbatim): "When new information conflicts with existing content:
#   1. Check the dates — newer sources generally supersede older ones. 2. If genuinely contradictory,
#   note both positions with dates and sources." Hermes length rule: "Keep pages scannable — a wiki
#   page should be readable in 30 seconds. Split pages over 200 lines."
# See FIDELITY.md for the full verbatim quotes + the faithful-vs-simplified mapping.
# NOTE: the prompts deliberately do NOT name "Karpathy" or "Hermes". A model needn't have ever heard
# of those — the real skill works by giving the agent operative INSTRUCTIONS inline, so we reproduce
# those instructions verbatim-in-substance. This makes the experiment self-contained and model-agnostic
# (a free/older-cutoff model behaves the same). Attribution lives in the comment above + FIDELITY.md.
COMPILE_SYS = (
    "You are compiling a knowledge wiki. Read the source notes below, extract the key information, and "
    "integrate it into ONE wiki page that synthesizes all of them into clean encyclopedic prose. "
    "Preserve every specific fact: named entities, numbers, percentages, and findings. Keep the page "
    "scannable — readable in about 30 seconds. Output ONLY the page (no preamble)."
)
UPDATE_SYS = (
    "You maintain a knowledge wiki page. A NEW source has arrived. Update the page: add new information "
    "and update facts. When new information conflicts with the page, prefer the newer source (it "
    "supersedes the older); if genuinely contradictory, keep both positions with their sources. "
    "Preserve the specific facts already on the page (named entities, numbers, findings) unless "
    "directly superseded. Keep the page scannable (readable in ~30 seconds). Output ONLY the updated page."
)
JUDGE_SYS = "You are a strict fact-checker. Answer with a single word: YES or NO."
JUDGE_BATCH_SYS = ("You are a strict fact-checker. For each numbered claim, decide if the TEXT "
                   "directly supports it. Output ONLY one line per claim in the form 'N: YES' or "
                   "'N: NO'. No other text.")


class LLM:
    """Thin transformers wrapper. 4-bit by default so a 14B fits a 16GB T4."""
    def __init__(self, model_id, load_4bit=True, max_new=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.model_id = model_id
        self.max_new = max_new
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = dict(device_map="auto", torch_dtype=torch.float16)
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()

    def gen(self, system, user, max_new=None, temp=0.0):
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Qwen3 supports a 'thinking' mode; disable for speed/determinism.
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        do_sample = temp and temp > 0
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new or self.max_new,
                                      do_sample=do_sample, temperature=(temp or 1.0),
                                      top_p=0.9 if do_sample else None,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][ids["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class APILLM:
    """OpenAI-compatible client (Kilo Code, Groq, OpenRouter, Google AI Studio, ...).

    Set base_url + key. Kilo Code free:
      base_url = https://api.kilo.ai/api/gateway   key = $KILO_API_KEY   model = kilo/auto-free
    """
    def __init__(self, model, base_url=None, api_key=None, max_new=512, timeout=120):
        import os as _os
        self.model = model
        self.model_id = model
        self.base_url = (base_url or _os.environ.get("LLM_BASE_URL")
                         or "https://api.kilo.ai/api/gateway").rstrip("/")
        self.api_key = api_key or _os.environ.get("LLM_API_KEY") or _os.environ.get("KILO_API_KEY")
        self.max_new = max_new
        self.timeout = timeout
        assert self.api_key, "set LLM_API_KEY (or KILO_API_KEY) env var"

    def gen(self, system, user, max_new=None, temp=0.0):
        import requests, time as _t
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "temperature": float(temp),
                   "max_tokens": max_new or self.max_new,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(6):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    _t.sleep(min(2 ** attempt, 30)); continue
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                # reasoning models put thinking in 'reasoning'; the answer is in 'content'
                return (msg.get("content") or "").strip()
            except Exception:
                if attempt == 5:
                    raise
                _t.sleep(min(2 ** attempt, 30))


def make_llm(backend, model, base_url=None, api_key=None):
    if backend == "api":
        return APILLM(model, base_url=base_url, api_key=api_key)
    return LLM(model, load_4bit=True)


def _last_yesno(text):
    """Robust YES/NO read: take the last standalone YES/NO token (after any reasoning)."""
    import re
    m = re.findall(r"\b(YES|NO)\b", (text or "").upper())
    return bool(m) and m[-1] == "YES"


def supported(llm, page, claim):
    """Judge: is the claim still supported by this page? (single-claim retention check).
    max_new is generous because reasoning models spend tokens 'thinking' before the answer."""
    user = f"TEXT:\n{page}\n\nCLAIM: {claim}\n\nIs the CLAIM directly supported by the TEXT? Answer YES or NO."
    return _last_yesno(llm.gen(JUDGE_SYS, user, max_new=384, temp=0.0))


def supported_batch(llm, page, claims):
    """Judge all claims on a page in ONE call -> {claim: bool}. ~3x fewer API calls.

    Missing/unparseable entries fall back to a single-claim check (so parse errors
    don't silently fabricate loss)."""
    import re
    if not claims:
        return {}
    if len(claims) == 1:
        return {claims[0]: supported(llm, page, claims[0])}
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    user = (f"TEXT:\n{page}\n\nCLAIMS:\n{numbered}\n\n"
            f"For each numbered claim, is it directly supported by the TEXT? "
            f"Answer one line per claim in the form '1: YES' or '1: NO'.")
    # generous budget: reasoning tokens + one short line per claim
    out = llm.gen(JUDGE_BATCH_SYS, user, max_new=64 * len(claims) + 256, temp=0.0).upper()
    verdict = {}
    # primary: "1: YES" / "1. ... YES" (tolerates junk between index and verdict on the line)
    for m in re.finditer(r"(?m)^\D*?(\d+)\s*[:.)\-][^\n]*?\b(YES|NO)\b", out):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(claims) and claims[idx] not in verdict:
            verdict[claims[idx]] = (m.group(2) == "YES")
    # fallback: assign YES/NO lines in order (handles models that drop the index, e.g. 'N: YES')
    if len(verdict) < len(claims):
        yn = [bool(re.search(r"\bYES\b", l)) for l in out.splitlines() if re.search(r"\b(YES|NO)\b", l)]
        for i, c in enumerate(claims):
            if c not in verdict and i < len(yn):
                verdict[c] = yn[i]
    # last resort: single-claim judge for anything still missing
    return {c: (verdict[c] if c in verdict else supported(llm, page, c)) for c in claims}


def build_pages(llm, groups):
    """Build one wiki page per group of source notes, INCREMENTALLY: seed the page from the
    first note, then fold in each remaining note one at a time via the same UPDATE_SYS step
    used in the maintenance rounds. This avoids a single one-shot compile of all notes at once
    (which was found to itself lose ~half the facts in one jump, before any rewriting even
    happens) so t=0 reflects normal incremental usage, not a compression artifact."""
    pages = []
    t0 = time.time()
    for gi, notes in enumerate(groups):
        page = llm.gen(COMPILE_SYS, f"[Source 1] {notes[0]}", max_new=1200, temp=0.7)
        for note in notes[1:]:
            user = f"CURRENT PAGE:\n{page}\n\nNEW SOURCE NOTE:\n{note}"
            page = llm.gen(UPDATE_SYS, user, max_new=1200, temp=0.7)
        pages.append(page)
        print(f"[build] page {gi+1}/{len(groups)} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    return pages


def build_pages_oneshot(llm, groups):
    """Fresh-rebuild control: compile each page from ALL its source notes in a SINGLE call
    (no rewrite history at all). This is the true ceiling for the rebuild comparison — it must
    stay one-shot (not the incremental build_pages() above) or it's just as lossy as t=0 and
    can no longer isolate 'rewriting damage' from 'data is fine'."""
    pages = []
    for notes in groups:
        body = "\n\n".join(f"[Source {i+1}] {t}" for i, t in enumerate(notes))
        pages.append(llm.gen(COMPILE_SYS, body, max_new=1200, temp=0.7))
    return pages


def run(arm, model_id, rounds, seed, group, small=False, out_dir=None,
        backend="hf", base_url=None, api_key=None):
    out_dir = out_dir or os.path.join(HERE, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    data = json.load(open(os.path.join(HERE, "data", "scifact_subset.json")))
    tracked, fillers = data["tracked"], data["fillers"]
    rng = random.Random(seed)
    rng.shuffle(fillers)

    # group tracked claims onto pages; each page also seeded with 2 fillers (realistic mix)
    pages_docs, claim_page, page_claims_of = [], {}, {}
    fi = 0
    for g in range(0, len(tracked), group):
        chunk = tracked[g:g + group]
        notes = [t["abstract"] for t in chunk] + [fillers[fi]["abstract"], fillers[fi + 1]["abstract"]]
        fi += 2
        pid = len(pages_docs)
        pages_docs.append(notes)
        page_claims_of[pid] = [t["claim"] for t in chunk]
        for t in chunk:
            claim_page[t["claim"]] = pid
    stream = fillers[fi:]  # remaining fillers feed the rewrite loop

    llm = make_llm(backend, model_id, base_url=base_url, api_key=api_key)
    t0 = time.time()
    pages = build_pages(llm, pages_docs)

    def retention():
        # one batched judge call per page (judges that page's tracked claims together)
        ok = 0
        for pid, cls in page_claims_of.items():
            v = supported_batch(llm, pages[pid], cls)
            ok += sum(1 for c in cls if v[c])
        return ok / len(tracked)

    log = [{"round": 0, "retention": retention()}]
    print(f"[{arm}] t=0 retention={log[0]['retention']:.3f}", flush=True)
    # checkpoint immediately so t=0 survives even if killed before round 1 finishes
    json.dump({"arm": arm, "model": model_id, "seed": seed, "rounds": rounds,
               "small": small, "log": log},
              open(os.path.join(out_dir, f"{arm}_{'small' if small else 'large'}_seed{seed}.json"), "w"))

    use_harness = (arm == "harness")
    si = 0
    for t in range(1, rounds + 1):
        for pid in range(len(pages)):
            if si >= len(stream):
                break
            new_doc = stream[si]["abstract"]; si += 1
            page_claims = page_claims_of[pid]
            before = supported_batch(llm, pages[pid], page_claims) if use_harness else None
            old = pages[pid]
            user = f"CURRENT PAGE:\n{pages[pid]}\n\nNEW SOURCE NOTE:\n{new_doc}"
            pages[pid] = llm.gen(UPDATE_SYS, user, max_new=1200, temp=0.7)  # room for reasoning + page
            if use_harness:
                after = supported_batch(llm, pages[pid], page_claims)
                # rollback if the rewrite destroyed any previously-supported fact
                if any(before[c] and not after[c] for c in page_claims):
                    pages[pid] = old
        r = retention()
        log.append({"round": t, "retention": r})
        print(f"[{arm}] t={t} retention={r:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        # checkpoint every round (survives Kaggle session cutoff)
        json.dump({"arm": arm, "model": model_id, "seed": seed, "rounds": rounds,
                   "small": small, "log": log},
                  open(os.path.join(out_dir, f"{arm}_{'small' if small else 'large'}_seed{seed}.json"), "w"))

    # fresh-rebuild control: recompile pages from all their current docs, no rewrite history
    rebuild_pages = build_pages_oneshot(llm, [pages_docs[p] for p in range(len(pages_docs))])
    saved = pages
    pages = rebuild_pages
    rebuild_ret = retention()
    pages = saved
    res = {"arm": arm, "model": model_id, "seed": seed, "rounds": rounds, "small": small,
           "log": log, "rebuild_retention": rebuild_ret,
           "r0": log[0]["retention"], "rT": log[-1]["retention"]}
    fn = os.path.join(out_dir, f"{arm}_{'small' if small else 'large'}_seed{seed}.json")
    json.dump(res, open(fn, "w"))
    print(f"=== DONE {arm} r0={res['r0']:.3f} rT={res['rT']:.3f} rebuild={rebuild_ret:.3f} -> {fn} ===", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "harness"], default="plain")
    ap.add_argument("--backend", choices=["hf", "api"], default="hf",
                    help="hf = local transformers (Kaggle); api = OpenAI-compatible (Kilo Code/Groq/...)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="hf: HF id (Qwen/Qwen3-8B); api: e.g. kilo/auto-free")
    ap.add_argument("--base_url", default=None, help="api base url (default Kilo: https://api.kilo.ai/api/gateway)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=5, help="tracked claims per wiki page")
    ap.add_argument("--small", action="store_true", help="label this as the small-model arm")
    args = ap.parse_args()
    run(args.arm, args.model, args.rounds, args.seed, args.group, small=args.small,
        backend=args.backend, base_url=args.base_url)
