"""E5 — two-clause contract (RETENTION + CURRENCY/SER) on a HoH temporal substrate,
maintained across a supersession update stream. Ties the certify stage's fidelity
certificate to the compile stage's staleness axis: the maintained contract certifies BOTH
  R >= 1 - delta   (current answer present)  and   SER <= eps   (stale answer absent).

t0: compile STALE-evidence corpus -> store asserts stale facts (high SER by design).
stream: current evidence supersedes stale (same title) in batches; store rewritten.
At each t, MAINTAINED policy re-audits only changed-page probes (+refresh) for BOTH
clauses. SER certificate = judge-noise-corrected UPPER bound on stale-answer rate.

Judge calibration for SER: TPR/FPR of the stale-answer judge estimated from t0
(stale present by construction) and from current-only contexts (stale absent).
External data only (HoH-QAs). Builder+judge Qwen2.5-14B vLLM :8102.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
from oai_client import VLLM
from stats import cp_lower, cp_upper, corrected_point, retention_lcb, ser_upper
from maintain import PageRouter

WORD_CAP = 350
CLEAN_SYS = (f"You are a wiki compiler. Write ONE wiki page (at most {WORD_CAP} words) that "
             "synthesizes ALL the facts below. PRESERVE every specific entity, name, "
             "quantity, and relationship. Dense declarative prose. No preamble.")
INCR_SYS = (f"You maintain a wiki page. Update it to reflect the NEW document, REPLACING "
            f"any superseded facts with the new ones, keeping other facts. Under {WORD_CAP} "
            f"words. Output ONLY the new page text.")
# answer-extraction judge: what does the store say the answer is?
ANS_SYS = ("Answer the question using ONLY the context. Reply with the shortest exact answer "
           "(a name or phrase). If the context does not contain the answer, reply UNKNOWN.")


def match(ans, target):
    a, t = ans.lower().strip(" .\"'"), target.lower().strip(" .\"'")
    return t in a or a in t if (a and a != "unknown") else False


def assign_pages(ids, per_page, seed):
    ids = list(ids); random.Random(seed).shuffle(ids)
    return [ids[i:i + per_page] for i in range(0, len(ids), per_page)]


def docs_text(corpus, mem):
    return "\n\n".join(f"[{corpus[d]['title']}] {corpus[d]['text']}" for d in mem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "e5_data.json"))
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--n_audit", type=int, default=120)  # rest of probes held out for calibration
    ap.add_argument("--refresh_k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results_e5.json"))
    args = ap.parse_args()

    data = json.load(open(args.data))
    corpus = {k: v for k, v in data["corpus0"].items()}
    json_corpus0 = data["corpus0"]  # frozen stale-evidence docs for calibration
    updates = data["updates"]
    probes = data["probes"]
    probe_doc = {f"p{k}": p for k, p in enumerate(probes)}
    members = assign_pages(corpus.keys(), args.docs_per_page, args.seed)
    page_of_doc = {d: pi for pi, m in enumerate(members) for d in m}
    rng = random.Random(args.seed)

    llm = VLLM()
    t0 = time.time()

    def gen_items(items, max_new_tokens, temperature):
        """Batched if engine supports it, else threaded."""
        if hasattr(llm, "gen_batch"):
            return llm.gen_batch(items, max_new_tokens=max_new_tokens, temperature=temperature)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(lambda it: llm.gen(it[0], it[1], max_new_tokens, temperature), items))

    pages = gen_items([(CLEAN_SYS, docs_text(corpus, mem)) for mem in members], 600, 0.7)
    print(f"built {len(pages)} pages; {len(probes)} probes", flush=True)

    def audit(pids, tag):
        items = [(ANS_SYS, f"CONTEXT:\n{pages[page_of_doc[pid]]}\n\nQUESTION: {probe_doc[pid]['question']}")
                 for pid in pids]
        ans = [a.strip() for a in gen_items(items, 32, 0.0)]
        ret = {pid: match(a, probe_doc[pid]["current_answer"]) for pid, a in zip(pids, ans)}
        stale = {pid: match(a, probe_doc[pid]["stale_answer"]) for pid, a in zip(pids, ans)}
        print(f"  {tag}: ret={sum(ret.values())}/{len(pids)} stale={sum(stale.values())}/{len(pids)}",
              flush=True)
        return ret, stale

    # Disjoint split: AUDIT probes (streamed + certified) vs CALIB probes (judge
    # TPR/FPR only). Calibration on held-out probes avoids the in-sample reuse where
    # the same probe estimates both the certificate and the judge correcting it.
    n_audit = min(args.n_audit, len(probes) - 20)
    pids = [f"p{k}" for k in range(n_audit)]
    calib_pids = [f"p{k}" for k in range(n_audit, len(probes))]
    ret, stale = audit(pids, "t0")

    # ---- judge calibration on the HELD-OUT calib probes, two constructed contexts ----
    # CURRENT context (current fact present, stale fact absent):
    #   retention-TPR = yields current ; SER-FPR = yields stale (false stale).
    # STALE context (stale fact present, current fact absent):
    #   retention-FPR = yields current (false current) ; SER-TPR = yields stale.
    sample = calib_pids

    cur_ans = [a.strip() for a in gen_items(
        [(ANS_SYS, f"CONTEXT:\n{updates[pid]['text']}\n\nQUESTION: {probe_doc[pid]['question']}")
         for pid in sample], 32, 0.0)]
    sta_ans = [a.strip() for a in gen_items(
        [(ANS_SYS, f"CONTEXT:\n{json_corpus0[pid]['text']}\n\nQUESTION: {probe_doc[pid]['question']}")
         for pid in sample], 32, 0.0)]
    ns = len(sample)
    ret_tpr = sum(match(a, probe_doc[pid]["current_answer"]) for pid, a in zip(sample, cur_ans))
    ret_fpr = sum(match(a, probe_doc[pid]["current_answer"]) for pid, a in zip(sample, sta_ans))
    ser_tpr = sum(match(a, probe_doc[pid]["stale_answer"]) for pid, a in zip(sample, sta_ans))
    ser_fpr = sum(match(a, probe_doc[pid]["stale_answer"]) for pid, a in zip(sample, cur_ans))
    calib = {"ret_tpr": (ret_tpr, ns), "ret_fpr": (ret_fpr, ns),
             "ser_tpr": (ser_tpr, ns), "ser_fpr": (ser_fpr, ns)}
    print(f"calib: ret TPR={ret_tpr}/{ns} FPR={ret_fpr}/{ns} | "
          f"SER TPR={ser_tpr}/{ns} FPR={ser_fpr}/{ns}", flush=True)

    def certs(ret_d, stale_d, n_for_cost):
        kR, nR = sum(ret_d.values()), len(ret_d)
        kS, nS = sum(stale_d.values()), len(stale_d)
        r_lcb, _ = retention_lcb(kR, nR, calib["ret_tpr"][0], calib["ret_tpr"][1],
                                 calib["ret_fpr"][0], calib["ret_fpr"][1], alpha=args.alpha)
        s_ub, _ = ser_upper(kS, nS, calib["ser_tpr"][0], calib["ser_tpr"][1],
                            calib["ser_fpr"][0], calib["ser_fpr"][1], alpha=args.alpha)
        return {"retention_raw": round(kR / nR, 3), "retention_LCB": round(r_lcb, 3),
                "SER_raw": round(kS / nS, 3), "SER_UB": round(s_ub, 3)}

    timeline = [{"t": 0, **certs(ret, stale, len(pids)), "cost": len(pids)}]
    print(f"t0 contract: {timeline[-1]}", flush=True)

    # ---- supersession stream ----
    router = PageRouter()
    order = list(range(n_audit))   # supersede only audited probes
    rng.shuffle(order)
    per_batch = max(1, len(order) // args.batches)
    maintained_ret, maintained_stale = dict(ret), dict(stale)
    cost_full, cost_maint = len(pids), len(pids)

    for t in range(1, args.batches + 1):
        batch = order[(t - 1) * per_batch: t * per_batch]
        changed = set()
        # group this batch's updates by target page; rewrites to DISTINCT pages are
        # independent -> batch them. A page with several updates absorbs ALL of them in
        # one rewrite (concatenated new documents).
        page_updates = {}
        for k in batch:
            pid = f"p{k}"
            page_updates.setdefault(page_of_doc[pid], []).append(pid)
        pis = list(page_updates)
        items = []
        for pi in pis:
            newdocs = "\n\n".join(f"[{updates[p]['title']}] {updates[p]['text']}"
                                  for p in page_updates[pi])
            items.append((INCR_SYS, f"CURRENT PAGE:\n{pages[pi]}\n\nNEW DOCUMENTS:\n{newdocs}"))
        new_texts = gen_items(items, 600, 0.7)
        for pi, txt in zip(pis, new_texts):
            pages[pi] = txt
            changed.add(pi)
        # FULL
        retF, staleF = audit(pids, f"t{t} FULL")
        cost_full += len(pids)
        # MAINTAINED: only changed-page probes + refresh
        need = [pid for pid in pids if page_of_doc[pid] in changed]
        untouched = [pid for pid in pids if pid not in need]
        refresh = rng.sample(untouched, min(args.refresh_k, len(untouched)))
        rU, sU = audit(need + refresh, f"t{t} MAINT({len(need)}+{len(refresh)})")
        maintained_ret.update(rU); maintained_stale.update(sU)
        cost_maint += len(need) + len(refresh)

        cF = certs(retF, staleF, len(pids))
        cM = certs(maintained_ret, maintained_stale, len(pids))
        timeline.append({"t": t, "n_superseded_cum": min(n_audit, t * per_batch),
                         "full": cF, "maintained": cM,
                         "cost_full": cost_full, "cost_maintained": cost_maint})
        print(f"t{t}: FULL ret={cF['retention_LCB']} SER_ub={cF['SER_UB']} | "
              f"MAINT ret={cM['retention_LCB']} SER_ub={cM['SER_UB']} | "
              f"cost {cost_maint}/{cost_full}", flush=True)

    json.dump({"config": vars(args), "calib": calib,
               "timeline": timeline, "n_calls": llm.n_calls,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=2)
    print("\n=== E5 DONE ===", flush=True)
    print(json.dumps({"t0": timeline[0], "final": timeline[-1]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
