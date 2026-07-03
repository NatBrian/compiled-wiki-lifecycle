"""Trained-maintainer stage: rejection-sampling SFT data for the faithful maintainer.

Improves on this stage's earlier diagnosis work (lora_datagen.py): k=8
candidates, more pages/chains, and it logs the SELECTION SIGNAL (mean best-of-k minus mean
random candidate retention) which is the Gate-1 evidence that a trainable signal exists
above judge noise.

ADAPTIVITY GUARD (pre-registered, a lesson carried over from the certify stage): no
evaluation/calibration probe text is ever seen here. Training docs are SciFact abstracts
disjoint from every eval/cal gold.
Candidate rewrites are scored against COMPILER-GENERATED per-document key-finding claims
("doc-claims") of the TRAINING docs only — never a benchmark claim.

mode=preserve  (default): candidate score = fraction of ALL page doc-claims retained.
mode=probe     : candidate score = fraction of a designated REWARDED subset (P_train) per
                 page retained (for the Goodhart-in-training experiment). Records which
                 claims were rewarded vs held so goodhart.py can measure the train/held gap.

Output: results/sft_<tag>.jsonl  {messages:[system,user,assistant]} + results/sft_<tag>.meta.json
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # this directory (self-contained, no nested code/)
CODE = os.path.dirname(os.path.dirname(HERE))      # .../compiled-wiki-lifecycle/code
ROOT = os.path.dirname(CODE)                       # .../compiled-wiki-lifecycle (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
import data as D
from oai_client import VLLM
from certify import CLEAN_SYS, JUDGE_SYS, doc_text, split_claims, assign_pages

WORD_CAP = 350  # default; overridable per run via --word_cap (tight cap = hard budget pressure)


def incr_sys(word_cap):
    # the maintainer prompt the policy is distilled FOR (must match deploy-time vanilla prompt)
    return (f"You maintain a wiki page. Rewrite the page to incorporate the key factual "
            f"findings of the NEW document (entities, quantities, directions of effect, "
            f"comparisons) while keeping the important facts already on the page. "
            f"The rewritten page must stay under {word_cap} words. Output ONLY the new page text.")


CLAIM_SYS = ("State the single most important specific factual finding of this scientific "
             "abstract in ONE self-contained sentence (name the entities and the direction "
             "or size of the effect). Output only that sentence.")
CLAIM_SYS2 = ("State a SECOND, different specific factual detail from this scientific abstract "
              "in ONE self-contained sentence (a distinct number, entity, method, or "
              "comparison not the headline finding). Output only that sentence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  # external dataset, not bundled in this repo -- fetch SciFact-Open
                    default=os.path.join(ROOT, "benchmarks", "scifact-open", "data"))
    ap.add_argument("--mode", choices=["preserve", "probe"], default="preserve")
    ap.add_argument("--n_pages", type=int, default=160)
    ap.add_argument("--chain_len", type=int, default=4)
    ap.add_argument("--k_cand", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--reward_frac", type=float, default=0.5,
                    help="mode=probe: fraction of each page's doc-claims that are REWARDED "
                         "(P_train); the rest are held-out (P_held)")
    ap.add_argument("--word_cap", type=int, default=WORD_CAP,
                    help="page word budget. Tight cap (e.g. 120) = hard budget pressure that "
                         "forces a keep/drop tradeoff, the regime where the adaptivity trap appears.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--tag", default="preserve")
    args = ap.parse_args()
    INCR_SYS = incr_sys(args.word_cap)
    word_cap = args.word_cap
    out_jsonl = os.path.join(P4, "results", f"sft_{args.tag}.jsonl")
    out_meta = os.path.join(P4, "results", f"sft_{args.tag}.meta.json")

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, _ = split_claims(claims_all, corpus_all, 40, 0)
    excluded = {g for c in cal_claims + audit_claims for g in c["all_golds"]}
    pool = [i for i in corpus_all if i not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    need = args.n_pages * (args.docs_per_page + args.chain_len)
    pool = pool[:need]
    print(f"mode={args.mode} training docs: {len(pool)} (excluded {len(excluded)} "
          f"eval/cal golds) k={args.k_cand}", flush=True)

    llm = VLLM(host=f"http://127.0.0.1:{args.port}")
    t0 = time.time()

    def gen_batch(items, mnt, temp):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(lambda it: llm.gen(it[0], it[1], mnt, temp), items))

    # 1. doc-claims (headline). For probe mode also a 2nd-detail claim per doc.
    claims = dict(zip(pool, [c.strip() for c in gen_batch(
        [(CLAIM_SYS, doc_text(corpus_all[d])) for d in pool], 80, 0.0)]))
    claims2 = {}
    if args.mode == "probe":
        claims2 = dict(zip(pool, [c.strip() for c in gen_batch(
            [(CLAIM_SYS2, doc_text(corpus_all[d])) for d in pool], 80, 0.0)]))
    print(f"doc-claims: {len(claims)} (+{len(claims2)} second)", flush=True)

    # 2. initial pages
    page_docs = [pool[i * args.docs_per_page:(i + 1) * args.docs_per_page]
                 for i in range(args.n_pages)]
    stream_docs = pool[args.n_pages * args.docs_per_page:]
    pages = gen_batch([(CLEAN_SYS, "\n\n".join(
        f"[Doc {k+1}] {doc_text(corpus_all[d])}" for k, d in enumerate(m)))
        for m in page_docs], 600, 0.7)
    print(f"initial pages: {len(pages)}", flush=True)

    def judge_present(page, claim_list):
        items = [(JUDGE_SYS, f"CONTEXT:\n{page}\n\nCLAIM: {cl}") for cl in claim_list]
        outs = gen_batch(items, 4, 0.0)
        return [1 if a.strip().upper().startswith("YES") else 0 for a in outs]

    # mode=probe: per-page reward mask over member doc-claims (P_train vs P_held)
    rewarded = {}   # pi -> set(doc_ids whose headline claim is rewarded)

    rows, sel_gaps, kept_total = [], [], 0
    for step in range(args.chain_len):
        new_for_page = {}
        for pi in range(len(pages)):
            if step * len(pages) + pi < len(stream_docs) and pages[pi]:
                idx = step * len(pages) + pi
                if idx < len(stream_docs):
                    new_for_page[pi] = stream_docs[idx]
        users = {pi: (f"CURRENT PAGE:\n{pages[pi]}\n\nNEW DOCUMENT:\n"
                      f"{doc_text(corpus_all[d])}") for pi, d in new_for_page.items()}
        cands = {pi: [] for pi in users}
        for _ in range(args.k_cand):
            outs = gen_batch([(INCR_SYS, users[pi]) for pi in users], 600, 0.9)
            for pi, o in zip(users, outs):
                cands[pi].append(o)

        for pi in users:
            members = page_docs[pi] + [new_for_page[pi]]
            if args.mode == "probe" and pi not in rewarded:
                rng.shuffle(members)
                n_rw = max(1, int(round(args.reward_frac * len(members))))
                rewarded[pi] = set(members[:n_rw])
            score_ids = (list(rewarded[pi]) if args.mode == "probe" else members)
            # score every candidate on the (rewarded) claim set
            cand_scores = []
            for cand in cands[pi]:
                ys = judge_present(cand, [claims[d] for d in score_ids])
                wpen = 0.02 if len(cand.split()) > word_cap * 1.15 else 0.0
                cand_scores.append(sum(ys) / max(1, len(ys)) - wpen)
            best = max(range(len(cand_scores)), key=lambda i: cand_scores[i])
            worst = min(cand_scores)
            sel_gaps.append(max(cand_scores) - sum(cand_scores) / len(cand_scores))
            if cand_scores[best] - worst >= args.margin:
                rows.append({"messages": [
                    {"role": "system", "content": INCR_SYS},
                    {"role": "user", "content": users[pi]},
                    {"role": "assistant", "content": cands[pi][best]}]})
                kept_total += 1
            pages[pi] = cands[pi][best]
            page_docs[pi] = members
        print(f"chain {step}: kept {kept_total} total, "
              f"mean best-minus-avg selection gap={sum(sel_gaps)/max(1,len(sel_gaps)):.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"mode": args.mode, "n_rows": len(rows), "k_cand": args.k_cand,
            "n_pages": args.n_pages, "chain_len": args.chain_len, "word_cap": word_cap,
            "selection_gap_mean": round(sum(sel_gaps) / max(1, len(sel_gaps)), 4),
            "selection_gap_n": len(sel_gaps), "margin": args.margin,
            "reward_frac": args.reward_frac if args.mode == "probe" else None,
            "rewarded_by_page": ({str(k): sorted(v) for k, v in rewarded.items()}
                                 if args.mode == "probe" else None),
            "claims": {str(k): v for k, v in claims.items()},
            "claims2": {str(k): v for k, v in claims2.items()} if claims2 else None,
            "minutes": round((time.time() - t0) / 60, 1), "n_calls": llm.n_calls}
    json.dump(meta, open(out_meta, "w"), indent=1)
    print(f"=== {len(rows)} SFT rows -> {out_jsonl} | sel_gap="
          f"{meta['selection_gap_mean']} | {llm.n_calls} calls, "
          f"{(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
