"""LossGate stage: Loss-Bounded Maintenance: gate-and-rollback wrapper with a
per-batch certified retention loss bound.

Extends trained-maintainer/'s constant-corpus maintenance harness
(p4_protocol.py) with a LossGate:
before committing each page rewrite, re-extract the facts the touched page supports
(GATE probe pool) and ROLL BACK any rewrite that destroys a previously-confirmed fact.
Each batch emits a certified per-transition bound

    R(t) >= R(t-1) - b_t   at confidence 1-alpha,

where b_t = Clopper-Pearson upper bound on the destroyed fraction among the facts
confirmed at t-1 (PAIRED flip count on the GATE pool). Because b_t is computed on the
GATE pool and the certificate is *validated* on a DISJOINT VALID pool the gate never
sees, coverage of the bound is a genuine test (E1), not a self-referential one.

Arms (--arm):
  vanilla              the rewrite-collapse disease baseline (rewrite, no gate)
  conservative         the maintain stage's cheapest cure (minimal-edit prompt, no gate)
  anchored             the maintain stage's strongest prompt, no gate
  lossgate_vanilla     vanilla rewrite + gate
  lossgate_conservative conservative rewrite + gate
  lossgate_anchored    anchored rewrite + gate

Make-or-break (A1): lossgate must show LOWER destroyed-retention than conservative
WHILE preserving currency (incorporation rate comparable to vanilla) -- i.e. it must
not win by freezing rewrites. Pre-registered kill -> pivot to R08.

Reuses: Store/PageRouter (maintain.py), CLEAN_SYS/judge_many/split_claims (certify.py),
retention_lcb/corrected_point/cp_upper (stats.py), shared calibration (contract/results_e1.json).
External data only (SciFact-Open). Builder+judge: Qwen2.5-14B local vLLM.
"""
import argparse, json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
CODE = os.path.dirname(os.path.dirname(HERE))      # .../llm-wiki/code
ROOT = os.path.dirname(CODE)                       # .../llm-wiki (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
import data as D
from oai_client import VLLM
from certify import CLEAN_SYS, JUDGE_SYS, doc_text, split_claims, assign_pages, judge_many, judge
from maintain import Store, PageRouter
from stats import retention_lcb, corrected_point, cp_upper, cp_lower

WORD_CAP = 350

VANILLA = (
    f"You maintain a wiki page. Rewrite the page to incorporate the key factual "
    f"findings of the NEW document (entities, quantities, directions of effect, "
    f"comparisons) while keeping the important facts already on the page. "
    f"The rewritten page must stay under {WORD_CAP} words. Output ONLY the new page text.")
CONSERVATIVE = (
    f"You maintain a wiki page. A NEW document has arrived. Make the SMALLEST possible "
    f"change to the page so it also captures the new document's key factual findings. "
    f"Rules: (1) APPEND the new findings; do NOT rewrite, re-summarize, or rephrase "
    f"sentences that are already on the page. (2) NEVER delete or shorten an existing "
    f"factual statement unless the new document DIRECTLY CONTRADICTS it — then replace "
    f"only the contradicted fact and keep everything else. (3) Keep the page under "
    f"{WORD_CAP} words; only if you would exceed it, compress the single least-specific "
    f"sentence. Output ONLY the new page text.")
ANCHORED = (
    f"You maintain a wiki page that records distinct factual claims. Treat every existing "
    f"claim as ANCHORED: reproduce it verbatim. Add the NEW document's key findings as "
    f"additional claims. Only edit an existing claim if the new document directly "
    f"contradicts it, changing only that one claim. Do not paraphrase, merge, summarize, "
    f"or drop existing claims. Stay under {WORD_CAP} words; if full, drop only the least "
    f"specific NEW claim rather than any existing one. Output ONLY the new page text.")
BASE_PROMPTS = {"vanilla": VANILLA, "conservative": CONSERVATIVE, "anchored": ANCHORED}

INCORP_CLAIM_SYS = ("State the single most important specific factual finding of this scientific "
                    "abstract in ONE self-contained sentence (name the entities and the direction "
                    "or size of the effect). Output only that sentence.")


def judge_pairs(llm, pairs, workers, tag):
    """pairs=[(context, claim)] -> list[0/1]."""
    return judge_many(llm, pairs, workers, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["vanilla", "conservative", "anchored",
                             "lossgate_vanilla", "lossgate_conservative", "lossgate_anchored"])
    ap.add_argument("--tau", type=int, default=0,
                    help="gate tolerance: rollback a rewrite if it destroys > tau confirmed facts")
    ap.add_argument("--maintainer_model", default=None,
                    help="route ONLY rewrites to this served model (e.g. LoRA adapter name)")
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--data",  # external dataset, not bundled in this repo -- fetch SciFact-Open
                    default=os.path.join(ROOT, "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs0", type=int, default=200)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_fillers", type=int, default=40)
    ap.add_argument("--replace_fillers", type=int, default=1)
    ap.add_argument("--gate_split", type=float, default=0.5, help="fraction of probes -> GATE pool")
    ap.add_argument("--split_seed", type=int, default=0, help="FIXED across arms for comparability")
    ap.add_argument("--track_incorp", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--cal_file", default=os.path.join(P2, "results_e1.json"))
    ap.add_argument("--max_incr", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gated = args.arm.startswith("lossgate_")
    base_arm = args.arm.split("lossgate_")[-1] if gated else args.arm
    incr_sys = BASE_PROMPTS[base_arm]

    # ---- data + probe set (split seed 0 fixed -> matches the certify stage's and the
    #      maintain stage's earlier calibration) ----
    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    _, audit_claims, _ = split_claims(claims_all, corpus_all, 40, 0)
    seen = set()
    probes = [c for c in audit_claims if not (c["claim"] in seen or seen.add(c["claim"]))]

    # disjoint GATE / VALID split (FIXED split_seed so every arm sees the same partition)
    sp = random.Random(args.split_seed)
    order = list(range(len(probes)))
    sp.shuffle(order)
    n_gate = int(round(len(probes) * args.gate_split))
    gate_idx = set(order[:n_gate])
    gate_probes = [probes[i] for i in order[:n_gate]]
    valid_probes = [probes[i] for i in order[n_gate:]]

    gold0 = {g for c in probes for g in c["all_golds"]}
    rng = random.Random(args.seed)
    fillers = [i for i in corpus_all if i not in gold0]
    rng.shuffle(fillers)
    n_fill0 = max(0, args.n_docs0 - len(gold0))
    corpus0 = {i: corpus_all[i] for i in list(gold0) + fillers[:n_fill0]}
    stream = fillers[n_fill0:]
    print(f"arm={args.arm} gated={gated} tau={args.tau} seed={args.seed} "
          f"probes={len(probes)} (gate={len(gate_probes)} valid={len(valid_probes)}) "
          f"corpus0={len(corpus0)} stream={len(stream)}", flush=True)

    # ---- judge calibration (shared with the certify stage and the maintain stage's
    #      other tracks) ----
    e1 = json.load(open(args.cal_file))
    vv = e1["verdicts"]
    cal = {"k_tpr": sum(vv["tpr_single"]), "n_tpr": len(vv["tpr_single"]),
           "k_fpr": sum(vv["fpr_wiki"]), "n_fpr": len(vv["fpr_wiki"])}
    tpr, fpr = cal["k_tpr"] / cal["n_tpr"], cal["k_fpr"] / cal["n_fpr"]

    def metrics(v):
        k, n = sum(v.values()), len(v)
        lcb, _ = retention_lcb(k, n, cal["k_tpr"], cal["n_tpr"], cal["k_fpr"], cal["n_fpr"],
                               alpha=args.alpha)
        return {"raw": round(k / n, 4), "corrected": round(corrected_point(k / n, tpr, fpr), 4),
                "lcb": round(lcb, 4), "k": k, "n": n}

    llm = VLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model)
    maint_llm = (VLLM(host=f"http://127.0.0.1:{args.port}", model=args.maintainer_model)
                 if args.maintainer_model else llm)

    t0 = time.time()
    gen_calls = {"build": 0, "maint": 0, "gate_judge": 0, "audit_judge": 0}

    A = Store(llm, corpus0, args.docs_per_page, args.seed, args.workers, "A")
    gen_calls["build"] += len(A.pages)
    router = PageRouter()

    def judge_set(probe_set, tag, bucket):
        pairs = [(A.pages[A.page_of_doc[c["gold"]]], c["claim"]) for c in probe_set]
        ys = judge_many(llm, pairs, args.workers, tag)
        gen_calls[bucket] += len(pairs)
        return {c["claim"]: y for c, y in zip(probe_set, ys)}

    def ingest_gated(new_docs):
        """Batch maintenance: freeze pages, route all new docs, bucket by target page,
        rewrite each TOUCHED page ONCE (incorporating all its new docs), gate per page,
        parallel across pages (no router calls / no shared mutation inside the parallel
        section -> no race). Roll back a whole page rewrite that destroys >tau confirmed
        GATE facts. Returns (changed_pages, committed_doc_ids, n_rolled_docs)."""
        # 1) route every new doc on the FROZEN page snapshot, bucket by target page
        buckets = {}
        for did, doc in new_docs.items():
            pi = router.nearest(doc_text(doc), A.pages)
            buckets.setdefault(pi, []).append(did)

        # 2) per-page worker: candidate rewrite + gate decision (pure, returns a plan)
        def work(item):
            pi, dids = item
            old_page = A.pages[pi]
            gate_here = [c for c in gate_probes if A.page_of_doc[c["gold"]] == pi]
            block = "\n\n".join(f"[New Doc {k+1}] {doc_text(new_docs[d])}"
                                for k, d in enumerate(dids))
            cand = maint_llm.gen(incr_sys, f"CURRENT PAGE:\n{old_page}\n\nNEW DOCUMENTS:\n{block}",
                                 max_new_tokens=args.max_incr, temperature=0.7)
            accept, gj = True, 0
            if gated and gate_here:
                pre = [judge(llm, old_page, c["claim"]) for c in gate_here]
                post = [judge(llm, cand, c["claim"]) for c in gate_here]
                gj = 2 * len(gate_here)
                if sum(1 for a, b in zip(pre, post) if a == 1 and b == 0) > args.tau:
                    accept = False
            return pi, dids, cand, accept, gj

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            plans = list(ex.map(work, list(buckets.items())))

        # 3) apply deterministically (single-threaded mutation)
        changed, committed_ids, rolled = set(), [], 0
        for pi, dids, cand, accept, gj in plans:
            gen_calls["maint"] += 1
            gen_calls["gate_judge"] += gj
            if accept:
                A.pages[pi] = cand
                for d in dids:
                    A.corpus[d] = new_docs[d]; A.members[pi].append(d); A.page_of_doc[d] = pi
                changed.add(pi); committed_ids.extend(dids)
            else:
                rolled += len(dids)  # new docs NOT absorbed -> currency cost (freeze tradeoff)
        return changed, committed_ids, rolled

    def incorp_rate(new_doc_ids):
        sample = [d for d in new_doc_ids if d in A.page_of_doc][:args.track_incorp]
        if not sample:
            return None
        claims = [llm.gen(INCORP_CLAIM_SYS, doc_text(A.corpus[d]), 80, 0.0).strip() for d in sample]
        pairs = [(A.pages[A.page_of_doc[d]], cl) for d, cl in zip(sample, claims)]
        ys = judge_many(llm, pairs, args.workers, "incorp")
        gen_calls["audit_judge"] += len(pairs)
        return round(sum(ys) / len(ys), 4)

    # ---- t=0 ----
    vg = judge_set(gate_probes, "t0 GATE", "gate_judge")
    vv_ = judge_set(valid_probes, "t0 VALID", "audit_judge")
    timeline = [{"t": 0, "gate": metrics(vg), "valid": metrics(vv_),
                 "b_t": 0.0, "b_t_naive": 0.0, "n_committed": 0, "n_rolled": 0,
                 "incorp_rate": None, "store_words": sum(len(p.split()) for p in A.pages),
                 "gen_calls": dict(gen_calls)}]
    prev_gate = dict(vg)

    replaceable = [i for i in corpus0 if i not in gold0]
    for t in range(1, args.batches + 1):
        new_docs = {}
        for _ in range(args.batch_fillers):
            if stream:
                f = stream.pop()
                new_docs[f] = corpus_all[f]
        if args.replace_fillers:
            out_ids = rng.sample(replaceable, min(len(new_docs), len(replaceable)))
            for fid in out_ids:
                A.corpus.pop(fid, None)
                replaceable.remove(fid)
            replaceable += list(new_docs)
        changed, committed_ids, rolled = ingest_gated(new_docs)
        committed = len(committed_ids)
        ir = incorp_rate(committed_ids)

        # end-of-batch audit: GATE pool drives the certificate; VALID pool is ground truth
        vg = judge_set(gate_probes, f"t{t} GATE", "gate_judge")
        vv_ = judge_set(valid_probes, f"t{t} VALID", "audit_judge")

        # b_t: PAIRED destroyed count on GATE pool among facts confirmed at t-1
        conf_prev = [c["claim"] for c in gate_probes if prev_gate.get(c["claim"], 0) == 1]
        d = sum(1 for cl in conf_prev if vg.get(cl, 0) == 0)
        n = len(conf_prev)
        b_naive = cp_upper(d, n, args.alpha) if n else 0.0
        # judge-noise correction: divide the destroyed proportion by the detectable margin
        # (conservative; >= naive since tpr-fpr < 1). Carries calibration from the
        # certify stage and the maintain stage's earlier tracks.
        b_t = min(1.0, b_naive / (tpr - fpr)) if (tpr - fpr) > 0 else 1.0
        prev_gate = dict(vg)

        timeline.append({"t": t, "gate": metrics(vg), "valid": metrics(vv_),
                         "b_t": round(b_t, 4), "b_t_naive": round(b_naive, 4),
                         "destroyed_gate": d, "n_conf_prev": n,
                         "n_new_docs": len(new_docs), "n_changed": len(changed),
                         "n_committed": committed, "n_rolled": rolled,
                         "incorp_rate": ir, "store_words": sum(len(p.split()) for p in A.pages),
                         "gen_calls": dict(gen_calls)})
        print(f"[{args.arm}] t={t}: commit={committed} roll={rolled} incorp={ir} "
              f"b_t={b_t:.3f} (d={d}/{n}) valid_R={timeline[-1]['valid']['corrected']} "
              f"valid_lcb={timeline[-1]['valid']['lcb']}", flush=True)

    json.dump({"config": vars(args), "cal": cal,
               "gate_probes": [c["claim"] for c in gate_probes],
               "valid_probes": [c["claim"] for c in valid_probes],
               "timeline": timeline, "gen_calls": gen_calls,
               "n_calls": llm.n_calls + (maint_llm.n_calls if maint_llm is not llm else 0),
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=1)
    print(f"=== P5 {args.arm} seed {args.seed} DONE {(time.time()-t0)/60:.1f} min "
          f"build={gen_calls['build']} maint={gen_calls['maint']} "
          f"gate={gen_calls['gate_judge']} ===", flush=True)


if __name__ == "__main__":
    main()
