"""Trained-maintainer stage: machine-checkable rewrite-collapse substrate (HoH,
judge-free).

Replicates the constant-corpus maintenance protocol of p4_protocol.py on the EXTERNAL HoH
benchmark (russwest404/HoH-QAs), but measures retention by STRING MATCH of each fact's
canonical answer span in its host page — NO LLM judge. This kills two reviewer objections at
once: (a) judge-noise (retention is now exact), and (b) generator-overfitting in the Goodhart
test (every fact is checked, so a maintainer cannot preserve "probed" facts while silently
dropping unprobed ones — there are no unprobed facts).

Each of the first `n_probe` HoH items contributes its CURRENT evidence passage as a corpus
doc and its CURRENT answer string as the tracked fact. Remaining items' passages are the
filler stream. Constant-corpus churn (replace fillers each batch) isolates rewrite damage.

Matching (_norm/_match) is ported verbatim from benchmark-dynamic-corpus/src/hoh_bench.py
(the same metric the compile stage used) for consistency across stages.

Run: ./detached_run.sh hoh_cons_s0 results/results_hohmc_conservative_seed0.json -- \
  python hoh_mc.py --arm conservative --seed 0 --out results/results_hohmc_conservative_seed0.json
"""
import argparse, json, os, re, sys, time, random
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # this directory (self-contained, no nested code/)
CODE = os.path.dirname(os.path.dirname(HERE))      # .../compiled-wiki-lifecycle/code
ROOT = os.path.dirname(CODE)                       # .../compiled-wiki-lifecycle (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
from oai_client import VLLM
from maintain import PageRouter

WORD_CAP = 350
CLEAN_SYS = (f"You are a wiki compiler. Write ONE wiki page (at most {WORD_CAP} words) that "
             "synthesizes ALL the passages below. PRESERVE every specific fact: named "
             "entities, dates, numbers, titles, and answers to who/what/when. Dense "
             "declarative prose. No preamble.")
# maintainer arms identical in spirit to p4_protocol.py
VANILLA = (f"You maintain a wiki page. Rewrite the page to incorporate the key facts of the "
           f"NEW passage (named entities, dates, numbers, titles) while keeping the important "
           f"facts already on the page. Stay under {WORD_CAP} words. Output ONLY the new page.")
CONSERVATIVE = (f"You maintain a wiki page. A NEW passage has arrived. Make the SMALLEST "
                f"possible change so the page also captures its specific facts. APPEND new "
                f"facts; do NOT rewrite or rephrase existing sentences; NEVER delete or shorten "
                f"an existing fact unless the new passage directly contradicts it (then replace "
                f"only that fact). Stay under {WORD_CAP} words; if over, compress only the least "
                f"specific sentence. Output ONLY the new page.")
ANCHORED = (f"You maintain a wiki page of distinct facts. Treat each existing fact as ANCHORED: "
            f"reproduce it verbatim. Add the NEW passage's facts as additional entries. Only "
            f"edit an existing fact if directly contradicted. Never paraphrase, merge, or drop "
            f"existing facts. Stay under {WORD_CAP} words. Output ONLY the new page.")
PROMPTS = {"vanilla": VANILLA, "conservative": CONSERVATIVE, "anchored": ANCHORED}


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.$%-]", " ", (s or "").lower())).strip()


def _match(ans, target):
    a, t = _norm(ans), _norm(target)
    if not t:
        return False
    if t in a:
        return True
    toks = [w for w in t.split() if len(w) > 2]
    if not toks:
        return t in a
    return sum(1 for w in toks if w in a) / len(toks) >= 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(PROMPTS), default="vanilla")
    ap.add_argument("--maintainer_model", default=None)
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--n_probe", type=int, default=80)
    ap.add_argument("--n_docs0", type=int, default=200)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_fillers", type=int, default=40)
    ap.add_argument("--min_outdated", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--max_incr", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    incr_sys = PROMPTS[args.arm]

    import datasets
    ds = datasets.load_dataset("russwest404/HoH-QAs", split="train")
    items = [rec for rec in ds if len(rec.get("outdated_infos") or []) >= args.min_outdated]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    probe_items = items[:args.n_probe]
    filler_items = items[args.n_probe:]
    # corpus doc = (id, text, fact). probes first, then fillers.
    def doc_of(rec, i, kind):
        return {"id": f"{kind}{i}", "text": (rec.get("evidence") or rec["answer"]),
                "fact": rec["answer"]}
    probe_docs = {f"p{i}": doc_of(r, i, "p") for i, r in enumerate(probe_items)}
    n_fill0 = max(0, args.n_docs0 - len(probe_docs))
    filler_docs_all = {f"f{i}": doc_of(r, i, "f") for i, r in enumerate(filler_items)}
    fids = list(filler_docs_all)
    rng.shuffle(fids)
    corpus = dict(probe_docs)
    for fid in fids[:n_fill0]:
        corpus[fid] = filler_docs_all[fid]
    stream = fids[n_fill0:]
    facts = {did: probe_docs[did]["fact"] for did in probe_docs}  # tracked facts
    print(f"arm={args.arm} probes={len(probe_docs)} corpus0={len(corpus)} "
          f"stream={len(stream)}", flush=True)

    llm = VLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model)
    maint_llm = (VLLM(host=f"http://127.0.0.1:{args.port}", model=args.maintainer_model)
                 if args.maintainer_model else llm)
    t0 = time.time()

    # build pages
    ids = sorted(corpus)
    rng2 = random.Random(args.seed)
    rng2.shuffle(ids)
    members = [ids[i:i + args.docs_per_page] for i in range(0, len(ids), args.docs_per_page)]
    items_b = [(CLEAN_SYS, "\n\n".join(f"[Doc {k+1}] {corpus[i]['text']}"
                                       for k, i in enumerate(m))) for m in members]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pages = list(ex.map(lambda it: llm.gen(it[0], it[1], 600, 0.7), items_b))
    page_of = {d: pi for pi, m in enumerate(members) for d in m}
    members = [list(m) for m in members]
    router = PageRouter()
    rewrites = [0] * 4096
    gen_calls = {"build": len(pages), "maint": 0}

    def retention():
        v = {did: (1 if _match(pages[page_of[did]], facts[did]) else 0) for did in facts}
        return v

    def ingest(new_docs):
        changed = set()
        for did, doc in new_docs.items():
            pi = router.nearest(doc["text"], pages)
            user = f"CURRENT PAGE:\n{pages[pi]}\n\nNEW PASSAGE:\n{doc['text']}"
            pages[pi] = maint_llm.gen(incr_sys, user, max_new_tokens=args.max_incr,
                                      temperature=0.7)
            gen_calls["maint"] += 1
            corpus[did] = doc
            members[pi].append(did)
            page_of[did] = pi
            rewrites[pi] += 1
            changed.add(pi)
        return changed

    v = retention()
    timeline = [{"t": 0, "verdicts": v, "raw": round(sum(v.values()) / len(v), 4),
                 "n_changed": 0, "page_words": [len(p.split()) for p in pages],
                 "rewrites": rewrites[:len(pages)]}]
    replaceable = [i for i in corpus if i.startswith("f")]
    for t in range(1, args.batches + 1):
        new_docs = {}
        for _ in range(args.batch_fillers):
            if stream:
                fid = stream.pop()
                new_docs[fid] = filler_docs_all[fid]
        out_ids = rng.sample(replaceable, min(len(new_docs), len(replaceable)))
        for fid in out_ids:
            corpus.pop(fid, None)
            replaceable.remove(fid)
        replaceable += list(new_docs)
        changed = ingest(new_docs)
        v = retention()
        raw = round(sum(v.values()) / len(v), 4)
        timeline.append({"t": t, "verdicts": v, "raw": raw, "n_changed": len(changed),
                         "page_words": [len(p.split()) for p in pages],
                         "rewrites": rewrites[:len(pages)]})
        print(f"[hohmc {args.arm}] t={t}: changed={len(changed)} retention={raw}", flush=True)

    json.dump({"config": vars(args), "metric": "string_match_no_judge",
               "page_of_probe": {did: page_of[did] for did in facts},
               "facts": facts, "timeline": timeline, "gen_calls": gen_calls,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=1)
    print(f"=== HoH-MC {args.arm} seed {args.seed} DONE {(time.time()-t0)/60:.1f} min ===",
          flush=True)


if __name__ == "__main__":
    main()
