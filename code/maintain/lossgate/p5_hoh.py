"""LossGate stage: E6: judge-free replication of Loss-Bounded Maintenance on HoH.

Same gate-and-rollback wrapper as p5_lossgate.py, but retention is measured by STRING
MATCH (ported from trained-maintainer/hoh_mc.py / the compile stage's
hoh_bench.py) instead of an LLM judge.
Two consequences the paper needs:
  (1) the gate's re-extraction check is DETERMINISTIC (no judge calls in the gate), so
      b_t carries NO judge-noise correction -- it is an exact binomial CP upper bound;
  (2) certificate coverage here cannot be a judge artifact -> E1's validity is corroborated
      on a fully machine-checkable substrate.

Arms: vanilla|conservative|anchored and their lossgate_* variants. Gate rolls back a
rewrite that destroys (>tau) a previously-matched GATE fact. Certificate validated on a
disjoint VALID fact pool the gate never sees.

External data only (russwest404/HoH-QAs). Builder: Qwen2.5-14B local vLLM.
"""
import argparse, json, os, re, sys, time, random
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
CODE = os.path.dirname(os.path.dirname(HERE))      # .../llm-wiki/code
ROOT = os.path.dirname(CODE)                       # .../llm-wiki (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
from oai_client import VLLM
from maintain import PageRouter
from stats import cp_upper

WORD_CAP = 350
CLEAN_SYS = (f"You are a wiki compiler. Write ONE wiki page (at most {WORD_CAP} words) that "
             "synthesizes ALL the passages below. PRESERVE every specific fact: named "
             "entities, dates, numbers, titles, and answers to who/what/when. Dense "
             "declarative prose. No preamble.")
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
BASE_PROMPTS = {"vanilla": VANILLA, "conservative": CONSERVATIVE, "anchored": ANCHORED}


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
    ap.add_argument("--arm", required=True,
                    choices=["vanilla", "conservative", "anchored",
                             "lossgate_vanilla", "lossgate_conservative", "lossgate_anchored"])
    ap.add_argument("--tau", type=int, default=0)
    ap.add_argument("--maintainer_model", default=None)
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--n_probe", type=int, default=80)
    ap.add_argument("--gate_split", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_docs0", type=int, default=200)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_fillers", type=int, default=40)
    ap.add_argument("--min_outdated", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--max_incr", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gated = args.arm.startswith("lossgate_")
    base_arm = args.arm.split("lossgate_")[-1] if gated else args.arm
    incr_sys = BASE_PROMPTS[base_arm]

    import datasets
    ds = datasets.load_dataset("russwest404/HoH-QAs", split="train")
    items = [rec for rec in ds if len(rec.get("outdated_infos") or []) >= args.min_outdated]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    probe_items = items[:args.n_probe]
    filler_items = items[args.n_probe:]

    def doc_of(rec, i, kind):
        return {"id": f"{kind}{i}", "text": (rec.get("evidence") or rec["answer"]),
                "fact": rec["answer"]}
    probe_docs = {f"p{i}": doc_of(r, i, "p") for i, r in enumerate(probe_items)}
    n_fill0 = max(0, args.n_docs0 - len(probe_docs))
    filler_docs_all = {f"f{i}": doc_of(r, i, "f") for i, r in enumerate(filler_items)}
    fids = list(filler_docs_all); rng.shuffle(fids)
    corpus = dict(probe_docs)
    for fid in fids[:n_fill0]:
        corpus[fid] = filler_docs_all[fid]
    stream = fids[n_fill0:]
    facts = {did: probe_docs[did]["fact"] for did in probe_docs}

    # disjoint GATE/VALID split of tracked facts (fixed split_seed across arms)
    sp = random.Random(args.split_seed)
    pids = list(facts); sp.shuffle(pids)
    n_gate = int(round(len(pids) * args.gate_split))
    gate_ids = set(pids[:n_gate]); valid_ids = set(pids[n_gate:])
    print(f"arm={args.arm} gated={gated} probes={len(probe_docs)} "
          f"(gate={len(gate_ids)} valid={len(valid_ids)}) corpus0={len(corpus)} "
          f"stream={len(stream)}", flush=True)

    llm = VLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model)
    maint_llm = (VLLM(host=f"http://127.0.0.1:{args.port}", model=args.maintainer_model)
                 if args.maintainer_model else llm)
    t0 = time.time()

    ids = sorted(corpus); rng2 = random.Random(args.seed); rng2.shuffle(ids)
    members = [ids[i:i + args.docs_per_page] for i in range(0, len(ids), args.docs_per_page)]
    items_b = [(CLEAN_SYS, "\n\n".join(f"[Doc {k+1}] {corpus[i]['text']}"
                                       for k, i in enumerate(m))) for m in members]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pages = list(ex.map(lambda it: llm.gen(it[0], it[1], 600, 0.7), items_b))
    page_of = {d: pi for pi, m in enumerate(members) for d in m}
    members = [list(m) for m in members]
    router = PageRouter()
    gen_calls = {"build": len(pages), "maint": 0}

    def match_set(id_set):
        return {did: (1 if _match(pages[page_of[did]], facts[did]) else 0) for did in id_set}

    def ingest_gated(new_docs):
        """Batch maintenance (freeze pages, route, bucket by page, one rewrite per touched
        page, parallel across pages). String-match gate -> deterministic, no judge calls."""
        buckets = {}
        for did, doc in new_docs.items():
            pi = router.nearest(doc["text"], pages)
            buckets.setdefault(pi, []).append(did)

        def work(item):
            pi, dids = item
            old_page = pages[pi]
            gate_here = [g for g in gate_ids if page_of[g] == pi]
            block = "\n\n".join(f"[New Passage {k+1}] {new_docs[d]['text']}"
                                for k, d in enumerate(dids))
            cand = maint_llm.gen(incr_sys, f"CURRENT PAGE:\n{old_page}\n\nNEW PASSAGES:\n{block}",
                                 max_new_tokens=args.max_incr, temperature=0.7)
            accept = True
            if gated and gate_here:
                pre = [1 if _match(old_page, facts[g]) else 0 for g in gate_here]
                post = [1 if _match(cand, facts[g]) else 0 for g in gate_here]
                if sum(1 for a, b in zip(pre, post) if a == 1 and b == 0) > args.tau:
                    accept = False
            return pi, dids, cand, accept

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            plans = list(ex.map(work, list(buckets.items())))

        changed, committed_ids, rolled = set(), [], 0
        for pi, dids, cand, accept in plans:
            gen_calls["maint"] += 1
            if accept:
                pages[pi] = cand
                for d in dids:
                    corpus[d] = new_docs[d]; members[pi].append(d); page_of[d] = pi
                changed.add(pi); committed_ids.extend(dids)
            else:
                rolled += len(dids)
        return changed, committed_ids, rolled

    def snap(id_set):
        v = match_set(id_set)
        return {"raw": round(sum(v.values()) / len(v), 4), "k": sum(v.values()), "n": len(v)}, v

    g0, vg = snap(gate_ids); v0, vv = snap(valid_ids)
    timeline = [{"t": 0, "gate": g0, "valid": v0, "b_t": 0.0, "destroyed_gate": 0,
                 "n_conf_prev": 0, "n_committed": 0, "n_rolled": 0,
                 "incorp": None, "store_words": sum(len(p.split()) for p in pages)}]
    prev_gate = dict(vg)
    replaceable = [i for i in corpus if i.startswith("f")]
    for t in range(1, args.batches + 1):
        new_docs = {}
        for _ in range(args.batch_fillers):
            if stream:
                fid = stream.pop(); new_docs[fid] = filler_docs_all[fid]
        out_ids = rng.sample(replaceable, min(len(new_docs), len(replaceable)))
        for fid in out_ids:
            corpus.pop(fid, None); replaceable.remove(fid)
        replaceable += list(new_docs)
        changed, committed_ids, rolled = ingest_gated(new_docs)
        committed = len(committed_ids)
        # incorporation (currency): did just-committed new docs' fact land on host page?
        ir = (round(sum(1 for d in committed_ids if _match(pages[page_of[d]], new_docs[d]["fact"]))
                    / len(committed_ids), 4) if committed_ids else 0.0)
        gm, vg = snap(gate_ids); vm, vv = snap(valid_ids)
        conf_prev = [g for g in gate_ids if prev_gate.get(g, 0) == 1]
        d = sum(1 for g in conf_prev if vg.get(g, 0) == 0)
        n = len(conf_prev)
        b_t = cp_upper(d, n, args.alpha) if n else 0.0   # NO judge correction (string match exact)
        prev_gate = dict(vg)
        timeline.append({"t": t, "gate": gm, "valid": vm, "b_t": round(b_t, 4),
                         "destroyed_gate": d, "n_conf_prev": n, "n_new_docs": len(new_docs),
                         "n_changed": len(changed), "n_committed": committed, "n_rolled": rolled,
                         "incorp": ir, "store_words": sum(len(p.split()) for p in pages)})
        print(f"[hoh {args.arm}] t={t}: commit={committed} roll={rolled} incorp={ir} "
              f"b_t={b_t:.3f} (d={d}/{n}) gate_R={gm['raw']} valid_R={vm['raw']}", flush=True)

    json.dump({"config": vars(args), "metric": "string_match_no_judge",
               "gate_ids": sorted(gate_ids), "valid_ids": sorted(valid_ids), "facts": facts,
               "timeline": timeline, "gen_calls": gen_calls,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=1)
    print(f"=== P5-HoH {args.arm} seed {args.seed} DONE {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
