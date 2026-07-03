"""Recompute the E1 certificate from STORED verdicts (no GPU) with the audit fixes:
  (1) drop FPR-absent claims whose evidence (SUPPORT/CONTRADICT) is in the built corpus,
  (2) TPR from 40 distinct single-gold claims (not pooled n=80),
  (3) simulation-from-known-R coverage,
  (4) RAG recall@k (CPU bge-small).
The per-claim verdict arrays in results_e1.json are in the exact order of the
deterministic (seed-0) claim splits, so we can reconstruct identities and re-derive.
Writes results_e1_fixed.json.
"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from stats import retention_lcb, corrected_point, cp_lower, cp_upper, coverage_simulation

DATA = os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data")
e1 = json.load(open(os.path.join(HERE, "results_e1.json")))
cfg = e1["config"]
vv = e1["verdicts"]
alpha = cfg["alpha"]

corpus_all = D.load_corpus(os.path.join(DATA, "corpus_candidates.jsonl"))
claims_all = D.load_claims(os.path.join(DATA, "claims.jsonl"))

# --- replicate OLD split (the order the stored verdicts were produced in) ---
rng = random.Random(cfg["seed"])
sup, absent = [], []
for c in claims_all:
    golds = [d for d, ev in c.get("evidence", {}).items()
             if ev.get("label") == "SUPPORT" and d in corpus_all]
    if golds:
        sup.append({"claim": c["claim"], "gold": golds[0], "all_golds": golds})
    else:
        absent.append({"claim": c["claim"], "evidence_docs": list(c.get("evidence", {}).keys())})
rng.shuffle(sup); rng.shuffle(absent)
cal_claims, audit_claims = sup[:cfg["n_cal"]], sup[cfg["n_cal"]:]

# rebuild corpus exactly as the run did (seed 0)
rng2 = random.Random(cfg["seed"])
gold, seen = [], set()
for c in cal_claims + audit_claims:
    for g in c["all_golds"]:
        if g not in seen:
            gold.append(g); seen.add(g)
fillers = [i for i in corpus_all if i not in seen]
rng2.shuffle(fillers)
docs = gold + fillers[:max(0, cfg["n_docs"] - len(gold))]
corpus_ids = set(docs)

# OLD fpr verdicts were over absent[:n_fpr]; identify contaminated entries
old_absent = absent[:cfg["n_fpr"]]
keep = [i for i, c in enumerate(old_absent)
        if not (set(c["evidence_docs"]) & corpus_ids)]
n_contaminated = len(old_absent) - len(keep)

def clean(arr):
    return [arr[i] for i in keep]

fpr_wiki = clean(vv["fpr_wiki"]); fpr_raw = clean(vv["fpr_raw"])
tpr_single = vv["tpr_single"]      # n=40 distinct, single-gold
v_wiki, v_rag = vv["wiki"], vv["rag"]

print(f"contaminated FPR claims removed: {n_contaminated}; clean n_fpr={len(fpr_wiki)}")
print(f"TPR n={len(tpr_single)} (single-gold, distinct claims)")

rep = {"n_contaminated_removed": n_contaminated, "n_fpr_clean": len(fpr_wiki),
       "tpr_single": round(sum(tpr_single)/len(tpr_single), 3),
       "tpr_buried": round(sum(vv["tpr_buried"])/len(vv["tpr_buried"]), 3),
       "n_tpr_distinct": len(tpr_single),
       "fpr_wiki_clean": round(sum(fpr_wiki)/len(fpr_wiki), 3),
       "fpr_raw_clean": round(sum(fpr_raw)/len(fpr_raw), 3)}
kt, nt = sum(tpr_single), len(tpr_single)
for name, v, fpr in [("wiki", v_wiki, fpr_wiki), ("rag_dense", v_rag, fpr_raw)]:
    kf, nf = sum(fpr), len(fpr)
    lcb, parts = retention_lcb(sum(v), len(v), kt, nt, kf, nf, alpha=alpha)
    rhat = corrected_point(sum(v)/len(v), kt/nt, kf/nf)
    tpr, fprr = kt/nt, kf/nf
    cov = {round(R,2): round(coverage_simulation(R, tpr, fprr, len(v), nt, nf,
            alpha=alpha, seed=cfg["seed"]), 3) for R in [0.3,0.4,0.5,0.6,0.7,0.8]}
    rep[name] = {"raw": round(sum(v)/len(v),3), "corrected_R": round(rhat,3),
                 "certificate_LCB": round(lcb,3), "min_sim_coverage": round(min(cov.values()),3),
                 "sim_coverage": cov}
    print(f"{name}: raw={sum(v)/len(v):.3f} R={rhat:.3f} LCB={lcb:.3f} "
          f"min_cov={min(cov.values()):.3f}")

# recall@k (CPU)
from certify import DenseRetriever, doc_text
corpus = {i: corpus_all[i] for i in docs}
retr = DenseRetriever(corpus)
recall = sum(1 for c in audit_claims if c["gold"] in retr.top(c["claim"], cfg["k_rag"]))/len(audit_claims)
rep["rag_recall_at_k"] = round(recall, 3)
print(f"RAG recall@{cfg['k_rag']} = {recall:.3f}")

json.dump(rep, open(os.path.join(HERE, "results_e1_fixed.json"), "w"), indent=2)
print("\nwrote results_e1_fixed.json")
