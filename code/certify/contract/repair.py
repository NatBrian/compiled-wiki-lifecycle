"""E4 — certified repair: closing the loop WITHOUT the adaptivity trap.

A store fails its contract (LCB < R_min). Repair strategies:
  targeted   — recompile pages whose REPAIR-SET probes failed, with an explicit
               "preserve findings about <failed claims>" hint (test-aware, like
               WiCER-style loops).
  rebuild    — full clean recompile (lottery re-roll; may not help).
  union      — compile twice independently, store both page variants, probe
               against concatenation (2x build & store cost).

THE POINT (statistical, our delta vs WiCER): if the repair was GUIDED by the same
probes you re-certify on, the certificate is invalid (adaptive selection).
We demonstrate it: targeted repair guided by REPAIR probes, then certify two ways:
  (a) ADAPTIVE (invalid): re-audit the same REPAIR probes -> inflated LCB;
  (b) HONEST  (valid):    audit the untouched HOLDOUT probes -> true certificate.
Gap (a)-(b) = the overclaim a WiCER-style certified loop would ship.

Probe split: AUDIT 76 -> REPAIR 38 / HOLDOUT 38 (seeded).
Judge calibration reused from E1. External data only.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from certify import (CLEAN_SYS, JUDGE_SYS, doc_text, split_claims,
                     judge_many, assign_pages)
from stats import retention_lcb, corrected_point

REPAIR_SYS = (CLEAN_SYS + " PAY SPECIAL ATTENTION to preserving findings relevant to "
              "these topics: {topics}")


def build_pages(llm, corpus, members, workers, sys_prompt=CLEAN_SYS, topics_by_page=None):
    items = []
    for i, mem in enumerate(members):
        sp = sys_prompt
        if topics_by_page and topics_by_page.get(i):
            sp = REPAIR_SYS.format(topics="; ".join(topics_by_page[i][:6]))
        blob = "\n\n".join(f"[Doc {k+1}] {doc_text(corpus[d])}" for k, d in enumerate(mem))
        items.append((sp, blob))
    if hasattr(llm, "gen_batch"):
        return llm.gen_batch(items, max_new_tokens=600, temperature=0.7)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda it: llm.gen(it[0], it[1], 600, 0.7), items))


def audit(llm, pages, page_of_doc, probes, workers, tag):
    pairs = [(pages[page_of_doc[c["gold"]]], c["claim"]) for c in probes]
    return judge_many(llm, pairs, workers, tag)


def cert(v, cal, alpha):
    lcb, _ = retention_lcb(sum(v), len(v), cal["k_tpr"], cal["n_tpr"],
                           cal["k_fpr"], cal["n_fpr"], alpha=alpha)
    r = corrected_point(sum(v) / len(v), cal["k_tpr"] / cal["n_tpr"],
                        cal["k_fpr"] / cal["n_fpr"])
    return {"lcb": round(lcb, 4), "r_hat": round(r, 4), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs", type=int, default=400)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--n_cal", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cal_file", default=os.path.join(HERE, "results_e1.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_e4.json"))
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, _ = split_claims(claims_all, corpus_all, args.n_cal, args.seed)
    rng = random.Random(args.seed + 13)
    rng.shuffle(audit_claims)
    repair_set, holdout = audit_claims[:38], audit_claims[38:76]

    e1 = json.load(open(args.cal_file))
    vv = e1["verdicts"]
    cal = {"k_tpr": sum(vv["tpr_single"]) + sum(vv["tpr_buried"]),
           "n_tpr": len(vv["tpr_single"]) + len(vv["tpr_buried"]),
           "k_fpr": sum(vv["fpr_wiki"]), "n_fpr": len(vv["fpr_wiki"])}

    gold = {g for c in audit_claims[:76] for g in c["all_golds"]}
    fillers = [i for i in corpus_all if i not in gold]
    rng.shuffle(fillers)
    corpus = {i: corpus_all[i] for i in list(gold) + fillers[:max(0, args.n_docs - len(gold))]}
    members = assign_pages(corpus.keys(), args.docs_per_page, args.seed)
    page_of_doc = {d: pi for pi, m in enumerate(members) for d in m}

    llm = VLLM()
    t0 = time.time()
    out = {"config": vars(args)}

    print("initial build...", flush=True)
    pages0 = build_pages(llm, corpus, members, args.workers)
    v_rep0 = audit(llm, pages0, page_of_doc, repair_set, args.workers, "initial repair-set")
    v_hold0 = audit(llm, pages0, page_of_doc, holdout, args.workers, "initial holdout")
    out["initial"] = {"repair_set": cert(v_rep0, cal, args.alpha),
                      "holdout": cert(v_hold0, cal, args.alpha)}

    # ---- targeted repair guided by repair-set failures ----
    failed = [c for c, y in zip(repair_set, v_rep0) if y == 0]
    fail_pages = {}
    for c in failed:
        fail_pages.setdefault(page_of_doc[c["gold"]], []).append(c["claim"])
    print(f"targeted repair: {len(failed)} failed probes on {len(fail_pages)} pages", flush=True)
    pages_t = list(pages0)
    rebuilt = build_pages(llm, corpus, [members[i] for i in sorted(fail_pages)],
                          args.workers, topics_by_page={
                              j: fail_pages[i] for j, i in enumerate(sorted(fail_pages))})
    for j, i in enumerate(sorted(fail_pages)):
        pages_t[i] = rebuilt[j]
    n_build_targeted = len(fail_pages)

    v_rep1 = audit(llm, pages_t, page_of_doc, repair_set, args.workers, "targeted: repair-set re-audit")
    v_hold1 = audit(llm, pages_t, page_of_doc, holdout, args.workers, "targeted: holdout audit")
    out["targeted"] = {
        "n_pages_rebuilt": n_build_targeted,
        "adaptive_invalid_cert": cert(v_rep1, cal, args.alpha),   # same probes that guided repair
        "honest_cert": cert(v_hold1, cal, args.alpha),            # untouched probes
        "overclaim_lcb_gap": round(cert(v_rep1, cal, args.alpha)["lcb"]
                                   - cert(v_hold1, cal, args.alpha)["lcb"], 4),
    }
    print(f"targeted: adaptive={out['targeted']['adaptive_invalid_cert']} "
          f"honest={out['targeted']['honest_cert']}", flush=True)

    # ---- full rebuild (lottery re-roll) ----
    pages_r = build_pages(llm, corpus, members, args.workers)
    v_rep2 = audit(llm, pages_r, page_of_doc, repair_set, args.workers, "rebuild: repair-set")
    v_hold2 = audit(llm, pages_r, page_of_doc, holdout, args.workers, "rebuild: holdout")
    out["rebuild"] = {"n_pages_rebuilt": len(members),
                      "repair_set": cert(v_rep2, cal, args.alpha),
                      "holdout": cert(v_hold2, cal, args.alpha)}

    # ---- compile-twice-union (store both variants) ----
    pages_u2 = build_pages(llm, corpus, members, args.workers)
    union = [a + "\n\n" + b for a, b in zip(pages0, pages_u2)]
    v_rep3 = audit(llm, union, page_of_doc, repair_set, args.workers, "union: repair-set")
    v_hold3 = audit(llm, union, page_of_doc, holdout, args.workers, "union: holdout")
    out["union"] = {"n_pages_rebuilt": len(members), "store_size_factor": 2.0,
                    "repair_set": cert(v_rep3, cal, args.alpha),
                    "holdout": cert(v_hold3, cal, args.alpha)}

    out["n_calls"] = llm.n_calls
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== E4 ===", flush=True)
    print(json.dumps({k: out[k] for k in ("initial", "targeted", "rebuild", "union")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
