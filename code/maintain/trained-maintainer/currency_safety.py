"""Trained-maintainer stage: currency safety of the TRAINED maintainer (preservation
must not entrench).

Adapts this stage's earlier diagnosis work (e5_entrench.py): a maintainer
trained to PRESERVE facts could learn to resist legitimate supersession (entrenchment). We
run the same staggered-supersession protocol on HoH pairs (e5_data.json: 160 (stale,current)
probes + distractors), but route BOTH the churn rewrites AND the explicit-REPLACE
supersession rewrites through the chosen maintainer (--maintainer_model). Build +
answer-extraction stay on the base model. Compare displacement / entrenchment to the vanilla
baseline (that earlier diagnosis work found ~77% displaced, ~3% entrenched).

A trained maintainer that refuses updates (entrenchment up, displacement down) is the
preservation-currency tradeoff — itself a publishable finding. A null (rates ~ vanilla) means
faithful-maintenance training is currency-safe.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # this directory (self-contained, no nested code/)
CODE = os.path.dirname(os.path.dirname(HERE))      # .../llm-wiki/code
ROOT = os.path.dirname(CODE)                       # .../llm-wiki (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
from oai_client import VLLM
from maintain import PageRouter
from currency import CLEAN_SYS, INCR_SYS, ANS_SYS, match, assign_pages, docs_text

WORD_CAP = 350
CHURN_SYS = (f"You maintain a wiki page. Rewrite the page to incorporate the key facts "
             f"of the NEW document while keeping the important facts already on the page. "
             f"The rewritten page must stay under {WORD_CAP} words. Output ONLY the new page text.")


def load_churn_pool(exclude_titles, n_needed, seed):
    from datasets import load_dataset
    ds = load_dataset("russwest404/HoH-QAs", split="train")
    pool = []
    for rec in ds:
        title = (rec.get("document") or {}).get("title") or ""
        ev = (rec.get("evidence") or "").strip()
        if ev and title and title not in exclude_titles:
            pool.append({"title": title, "text": ev})
    random.Random(seed).shuffle(pool)
    seen, out = set(), []
    for d in pool:
        if d["title"] not in seen:
            seen.add(d["title"]); out.append(d)
        if len(out) >= n_needed:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(P2, "e5_data.json"))
    ap.add_argument("--maintainer_model", default=None,
                    help="route churn+replace rewrites here (None=vanilla base)")
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--n_audit", type=int, default=120)
    ap.add_argument("--churn_fillers", type=int, default=24)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.data))
    corpus = dict(data["corpus0"])
    updates = data["updates"]
    probes = data["probes"]
    probe_doc = {f"p{k}": p for k, p in enumerate(probes)}
    members = assign_pages(corpus.keys(), args.docs_per_page, args.seed)
    page_of_doc = {d: pi for pi, m in enumerate(members) for d in m}
    rng = random.Random(args.seed)
    titles_in = {v["title"] for v in corpus.values()} | {v["title"] for v in updates.values()}
    churn_pool = load_churn_pool(titles_in, args.batches * args.churn_fillers + 8, args.seed)
    print(f"corpus0={len(corpus)} pages={len(members)} churn_pool={len(churn_pool)} "
          f"maint={args.maintainer_model}", flush=True)

    llm = VLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model)
    maint = (VLLM(host=f"http://127.0.0.1:{args.port}", model=args.maintainer_model)
             if args.maintainer_model else llm)
    t0 = time.time()

    def gen_items(items, mnt, temp, eng=llm):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(lambda it: eng.gen(it[0], it[1], mnt, temp), items))

    pages = gen_items([(CLEAN_SYS, docs_text(corpus, m)) for m in members], 600, 0.7)
    rewrites = [0] * (len(pages) + 8)
    print(f"built {len(pages)} pages", flush=True)

    def audit(pids, tag):
        items = [(ANS_SYS, f"CONTEXT:\n{pages[page_of_doc[pid]]}\n\nQUESTION: "
                           f"{probe_doc[pid]['question']}") for pid in pids]
        ans = [a.strip() for a in gen_items(items, 32, 0.0)]
        out = {}
        for pid, a in zip(pids, ans):
            out[pid] = {"answer": a[:80],
                        "current": int(match(a, probe_doc[pid]["current_answer"])),
                        "stale": int(match(a, probe_doc[pid]["stale_answer"]))}
        print(f"  {tag}: current={sum(v['current'] for v in out.values())}/{len(pids)} "
              f"stale={sum(v['stale'] for v in out.values())}/{len(pids)}", flush=True)
        return out

    n_audit = min(args.n_audit, len(probes) - 20)
    pids = [f"p{k}" for k in range(n_audit)]
    order = pids[:]; rng.shuffle(order)
    per_batch = len(order) // args.batches
    cohorts = [order[i * per_batch:(i + 1) * per_batch] for i in range(args.batches)]
    cohorts[-1] += order[args.batches * per_batch:]

    router = PageRouter()
    churn_iter = iter(churn_pool)
    superseded, injections, resurrect = [], [], []

    for t in range(1, args.batches + 1):
        churned = set()
        for _ in range(args.churn_fillers):
            d = next(churn_iter, None)
            if d is None:
                break
            pi = router.nearest(f"[{d['title']}] {d['text']}", pages)
            user = f"CURRENT PAGE:\n{pages[pi]}\n\nNEW DOCUMENT:\n[{d['title']}] {d['text']}"
            pages[pi] = maint.gen(CHURN_SYS, user, 600, 0.7)
            rewrites[pi] += 1; churned.add(pi)
        batch_inj = []
        for pid in cohorts[t - 1]:
            pi = page_of_doc[pid]
            age, words = rewrites[pi], len(pages[pi].split())
            user = (f"CURRENT PAGE:\n{pages[pi]}\n\nNEW DOCUMENTS:\n"
                    f"[{updates[pid]['title']}] {updates[pid]['text']}")
            pages[pi] = maint.gen(INCR_SYS, user, 600, 0.7)
            rewrites[pi] += 1
            batch_inj.append({"pid": pid, "page": pi, "age_at_injection": age,
                              "page_words_at_injection": words, "batch": t})
        v_now = audit([b["pid"] for b in batch_inj], f"t{t} cohort")
        for b in batch_inj:
            b.update(v_now[b["pid"]])
            b["outcome"] = ("entrenched" if b["stale"] else
                            "displaced" if b["current"] else "neither")
        injections += batch_inj
        if superseded:
            v_old = audit(superseded, f"t{t} resurrection({len(superseded)})")
            resurrect.append({"t": t, "n_stale": sum(v["stale"] for v in v_old.values()),
                              "n_current": sum(v["current"] for v in v_old.values())})
        superseded += [b["pid"] for b in batch_inj]
        dis = sum(1 for b in batch_inj if b["outcome"] == "displaced")
        ent = sum(1 for b in batch_inj if b["outcome"] == "entrenched")
        print(f"t={t}: churned={len(churned)} injected={len(batch_inj)} "
              f"displaced={dis} entrenched={ent}", flush=True)

    n = len(injections)
    summary = {"maintainer_model": args.maintainer_model, "n_injections": n,
               "displacement_rate": round(sum(1 for b in injections if b["outcome"] == "displaced") / n, 4),
               "entrenchment_rate": round(sum(1 for b in injections if b["outcome"] == "entrenched") / n, 4),
               "neither_rate": round(sum(1 for b in injections if b["outcome"] == "neither") / n, 4)}
    json.dump({"config": vars(args), "summary": summary,
               "page_rewrites_final": rewrites[:len(pages)],
               "injections": injections, "resurrection": resurrect,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=1)
    print(f"=== CURRENCY-SAFETY seed {args.seed} maint={args.maintainer_model} "
          f"displaced={summary['displacement_rate']} entrenched={summary['entrenchment_rate']} ===",
          flush=True)


if __name__ == "__main__":
    main()
