"""E2+E3 — incremental certificate maintenance under streaming updates (+ lottery transfer).

Store starts from 60 audit-claims' gold docs (+fillers). T batches stream in: each batch
injects held-back gold docs (new certified facts) + filler docs, routed to the
embedding-nearest existing page, which is REWRITTEN (INCR_SYS) to absorb them.

Certificate policies compared at every t:
  STALE      — certificate issued at t=0, never re-audited (what products implicitly do).
  FULL       — full re-audit of all in-store probes each batch (gold standard, T*n cost).
  MAINTAINED — re-judge only probes whose page changed this batch (+ refresh sample);
               unchanged-page verdicts carry over (temp-0 judge => verdict is a function
               of page text). Cost ~ |write set|.

Truth proxy at t: corrected point estimate from the FULL policy's verdicts.
Validity: policy LCB <= truth. Also logs naive (uncorrected) variants.

E3: second independent build (B) at t=0; certificate computed from A's verdicts is
checked against B's corrected truth (transfer miscoverage = build lottery cost).

External data only (SciFact-Open). Builder+judge Qwen2.5-14B vLLM :8102.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from certify import (CLEAN_SYS, JUDGE_SYS, doc_text, split_claims, judge,
                     judge_many, DenseRetriever, assign_pages)
from stats import retention_lcb, corrected_point, cp_lower

WORD_CAP = 350
INCR_SYS = (f"You maintain a wiki page. Rewrite the page to incorporate the key factual "
            f"findings of the NEW document (entities, quantities, directions of effect, "
            f"comparisons) while keeping the important facts already on the page. "
            f"The rewritten page must stay under {WORD_CAP} words. Output ONLY the new page text.")


class Store:
    """Compiled wiki with page-level provenance and write tracking."""

    def __init__(self, llm, corpus, per_page, seed, workers, tag):
        self.llm = llm
        self.corpus = dict(corpus)
        self.per_page = per_page
        self.workers = workers
        members = assign_pages(corpus.keys(), per_page, seed)
        def one(mem):
            blob = "\n\n".join(f"[Doc {k+1}] {doc_text(corpus[i])}" for k, i in enumerate(mem))
            return llm.gen(CLEAN_SYS, blob, max_new_tokens=600, temperature=0.7)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            self.pages = list(ex.map(one, members))
        self.members = [list(m) for m in members]
        self.page_of_doc = {d: pi for pi, m in enumerate(members) for d in m}
        print(f"  built store {tag}: {len(self.pages)} pages", flush=True)

    def ingest_batch(self, new_docs, retr_pages):
        """Route each new doc to nearest page (by embedding over page texts), rewrite it.
        Returns set of changed page indices."""
        changed = set()
        # route sequentially (page texts change), rewrite per doc
        for did, doc in new_docs.items():
            pi = retr_pages.nearest(doc_text(doc), self.pages)
            user = f"CURRENT PAGE:\n{self.pages[pi]}\n\nNEW DOCUMENT:\n{doc_text(doc)}"
            self.pages[pi] = self.llm.gen(INCR_SYS, user, max_new_tokens=600, temperature=0.7)
            self.corpus[did] = doc
            self.members[pi].append(did)
            self.page_of_doc[did] = pi
            changed.add(pi)
        return changed


class PageRouter:
    """Embedding similarity of a doc against current page texts (CPU bge-small).
    Caches page embeddings; re-encodes only pages whose text changed."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        self._texts, self._embs = [], None

    def _sync(self, pages):
        stale = [i for i, p in enumerate(pages)
                 if i >= len(self._texts) or self._texts[i] != p]
        if stale:
            new = self.model.encode([pages[i] for i in stale], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=False)
            if self._embs is None or len(self._texts) < len(pages):
                import numpy as np
                embs = np.zeros((len(pages), new.shape[1]), dtype=new.dtype)
                if self._embs is not None:
                    embs[:len(self._embs)] = self._embs[:len(pages)]
                self._embs = embs
            for j, i in enumerate(stale):
                self._embs[i] = new[j]
            self._texts = list(pages)

    def nearest(self, text, pages):
        self._sync(pages)
        q = self.model.encode([text], normalize_embeddings=True)[0]
        return int(self.np.argmax(self._embs @ q))


def cert_from(verdicts_by_probe, probes_in, cal, alpha):
    v = [verdicts_by_probe[c["claim"]] for c in probes_in]
    lcb, parts = retention_lcb(sum(v), len(v), cal["k_tpr"], cal["n_tpr"],
                               cal["k_fpr"], cal["n_fpr"], alpha=alpha)
    p = sum(v) / max(1, len(v))
    return {"lcb": round(lcb, 4), "naive_lcb": round(cp_lower(sum(v), len(v), alpha), 4),
            "raw": round(p, 4), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs0", type=int, default=240)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--n_cal", type=int, default=40)
    ap.add_argument("--n_fpr", type=int, default=120)
    ap.add_argument("--n_hold_gold", type=int, default=16)  # audit golds injected later
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_fillers", type=int, default=24)
    ap.add_argument("--refresh_k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cal_file", default=os.path.join(HERE, "results_e1.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_e2e3.json"))
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, absent = split_claims(claims_all, corpus_all, args.n_cal, args.seed)
    rng = random.Random(args.seed + 7)

    # judge calibration reused from E1 (same judge, same construction)
    e1 = json.load(open(args.cal_file))
    vv = e1["verdicts"]
    cal = {"k_tpr": sum(vv["tpr_single"]) + sum(vv["tpr_buried"]),
           "n_tpr": len(vv["tpr_single"]) + len(vv["tpr_buried"]),
           "k_fpr": sum(vv["fpr_wiki"]), "n_fpr": len(vv["fpr_wiki"])}

    # split audit claims: 60 in-store at t0, n_hold_gold injected via stream
    rng.shuffle(audit_claims)
    hold = audit_claims[:args.n_hold_gold]
    base = audit_claims[args.n_hold_gold:]
    gold0 = {g for c in base for g in c["all_golds"]}
    gold_hold = {g for c in hold for g in c["all_golds"]}
    fillers = [i for i in corpus_all if i not in gold0 | gold_hold]
    rng.shuffle(fillers)
    n_fill0 = max(0, args.n_docs0 - len(gold0))
    corpus0 = {i: corpus_all[i] for i in list(gold0) + fillers[:n_fill0]}
    stream_fill = fillers[n_fill0:]

    llm = VLLM()
    t0 = time.time()
    print("building stores A and B...", flush=True)
    A = Store(llm, corpus0, args.docs_per_page, args.seed, args.workers, "A")
    B = Store(llm, corpus0, args.docs_per_page, args.seed, args.workers, "B")
    router = PageRouter()

    def judge_probes(store, probes, tag):
        pairs = [(store.pages[store.page_of_doc[c["gold"]]], c["claim"]) for c in probes]
        return dict(zip([c["claim"] for c in probes],
                        judge_many(llm, pairs, args.workers, tag)))

    # ---- t=0: full audit on A and B ----
    in_store = list(base)
    vA = judge_probes(A, in_store, "t0 audit A")
    vB = judge_probes(B, in_store, "t0 audit B")
    certA0 = cert_from(vA, in_store, cal, args.alpha)
    truthB0 = corrected_point(sum(vB.values()) / len(vB),
                              cal["k_tpr"] / cal["n_tpr"], cal["k_fpr"] / cal["n_fpr"])
    e3 = {"certA_lcb": certA0["lcb"], "truthB_corrected": round(truthB0, 4),
          "transfer_valid": bool(certA0["lcb"] <= truthB0),
          "per_probe_disagree": round(sum(vA[c["claim"]] != vB[c["claim"]]
                                          for c in in_store) / len(in_store), 4)}
    print(f"E3 lottery: certA={certA0['lcb']} truthB={truthB0:.3f} "
          f"disagree={e3['per_probe_disagree']}", flush=True)

    # ---- stream on A ----
    maintained = dict(vA)          # carried verdicts
    cost = {"stale": len(in_store), "full": len(in_store), "maintained": len(in_store)}
    hold_iter = iter(hold)
    timeline = [{"t": 0, "stale": certA0, "full": certA0,
                 "maintained": certA0, "truth": certA0, "in_store": len(in_store),
                 "cost": dict(cost)}]

    for t in range(1, args.batches + 1):
        inject = list(next_golds(hold_iter, 2))
        new_docs = {}
        for c in inject:
            for g in c["all_golds"]:
                if g not in A.corpus:
                    new_docs[g] = corpus_all[g]
        for _ in range(args.batch_fillers):
            if stream_fill:
                f = stream_fill.pop()
                new_docs[f] = corpus_all[f]
        changed = A.ingest_batch(new_docs, router)
        in_store = in_store + inject
        print(f"t={t}: +{len(new_docs)} docs, {len(changed)} pages rewritten, "
              f"probes={len(in_store)}", flush=True)

        # FULL: re-judge everything
        vFull = judge_probes(A, in_store, f"t{t} FULL")
        cost["full"] += len(in_store)
        truth = corrected_point(sum(vFull.values()) / len(vFull),
                                cal["k_tpr"] / cal["n_tpr"], cal["k_fpr"] / cal["n_fpr"])

        # MAINTAINED: re-judge write-set probes + new probes + refresh sample
        need = [c for c in in_store if A.page_of_doc[c["gold"]] in changed
                or c["claim"] not in maintained]
        untouched = [c for c in in_store if c not in need]
        refresh = rng.sample(untouched, min(args.refresh_k, len(untouched)))
        upd = judge_probes(A, need + refresh, f"t{t} MAINT({len(need)}+{len(refresh)})")
        maintained.update(upd)
        cost["maintained"] += len(need) + len(refresh)

        timeline.append({
            "t": t, "in_store": len(in_store), "n_new_docs": len(new_docs),
            "n_changed_pages": len(changed),
            "stale": certA0,
            "full": cert_from(vFull, in_store, cal, args.alpha),
            "maintained": cert_from(maintained, in_store, cal, args.alpha),
            "truth_corrected": round(truth, 4),
            "stale_valid": bool(certA0["lcb"] <= truth),
            "maintained_valid": bool(cert_from(maintained, in_store, cal, args.alpha)["lcb"] <= truth),
            "cost": dict(cost),
        })
        print(f"   truth={truth:.3f} stale_lcb={certA0['lcb']} "
              f"maint_lcb={timeline[-1]['maintained']['lcb']} "
              f"cost={cost}", flush=True)

    json.dump({"config": vars(args), "e3_lottery_transfer": e3, "cal": cal,
               "timeline": timeline,
               "n_calls": llm.n_calls, "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=2)
    print("\n=== E2/E3 DONE ===", flush=True)
    print(json.dumps({"e3": e3, "final": timeline[-1]}, indent=2), flush=True)


def next_golds(it, k):
    out = []
    for _ in range(k):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


if __name__ == "__main__":
    main()
