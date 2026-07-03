"""Trained-maintainer stage: Goodhart-in-training evaluation (adaptivity trap in the
memory loop).

Reconstructs the SAME training-style pages + chains used by sft_datagen.py --mode probe
(same data seed/exclusions), then REPLAYS each chain deterministically (temp 0, single
rewrite per step — no k-sampling) with a CHOSEN maintainer (prompt + optional adapter), and
measures final retention separately on three claim sets per page:

  P_train : the doc-claims that were REWARDED during the maintainer's training (from the
            probe-mode reward mask in sft_<tag>.meta.json).
  P_held  : the page's other doc-claims (same generator, same type, never rewarded).
  P_fresh : a regenerated second-detail claim per doc (CLAIM_SYS2), never seen in training
            (guards against "the maintainer just got generally better / leaked the generator").

The trap is real iff, after training, (P_train - P_held) retention gap is much larger than
under the vanilla maintainer replaying the same chains. Run once per maintainer (vanilla,
trained) and once per reward_frac for the dose-response.

Mechanism: also logs per-claim verbatim-token overlap of surviving vs dropped claims.
"""
import argparse, json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # this directory (self-contained, no nested code/)
CODE = os.path.dirname(os.path.dirname(HERE))      # .../llm-wiki/code
ROOT = os.path.dirname(CODE)                       # .../llm-wiki (repo root)
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
P2 = os.path.join(CODE, "certify", "contract")     # certify stage data assets (e5_data.json, results_e1.json)
import data as D
from oai_client import VLLM
from certify import CLEAN_SYS, JUDGE_SYS, doc_text, split_claims

WORD_CAP = 350  # default; eval replay cap, overridable via --word_cap (defaults to the meta's training cap)


def incr_vanilla(word_cap):
    return (f"You maintain a wiki page. Rewrite the page to incorporate the key factual "
            f"findings of the NEW document (entities, quantities, directions of effect, "
            f"comparisons) while keeping the important facts already on the page. "
            f"The rewritten page must stay under {word_cap} words. Output ONLY the new page text.")


def overlap(claim, page):
    ct = {w for w in claim.lower().split() if len(w) > 3}
    pt = set(page.lower().split())
    return round(len(ct & pt) / max(1, len(ct)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="results/sft_<tag>.meta.json from probe-mode datagen")
    ap.add_argument("--maintainer_model", default=None, help="adapter name (None=vanilla base)")
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--label", required=True, help="e.g. vanilla / trained_f50")
    ap.add_argument("--data",  # external dataset, not bundled in this repo -- fetch SciFact-Open
                    default=os.path.join(ROOT, "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_pages", type=int, default=None, help="default: from meta")
    ap.add_argument("--chain_len", type=int, default=None)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--word_cap", type=int, default=None,
                    help="eval-replay page budget; defaults to the meta's training word_cap")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    n_pages = args.n_pages or meta["n_pages"]
    chain_len = args.chain_len or meta["chain_len"]
    word_cap = args.word_cap or meta.get("word_cap", WORD_CAP)
    INCR_VANILLA = incr_vanilla(word_cap)
    rewarded_by_page = {int(k): set(v) for k, v in (meta["rewarded_by_page"] or {}).items()}
    claims = {int(k): v for k, v in meta["claims"].items()}
    claims2 = {int(k): v for k, v in (meta["claims2"] or {}).items()}

    # reconstruct the SAME doc pool as sft_datagen (identical seed/exclusions/shuffle)
    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, audit_claims, _ = split_claims(claims_all, corpus_all, 40, 0)
    excluded = {g for c in cal_claims + audit_claims for g in c["all_golds"]}
    pool = [i for i in corpus_all if i not in excluded]
    random.Random(args.seed).shuffle(pool)
    need = n_pages * (args.docs_per_page + chain_len)
    pool = pool[:need]
    page_docs = [pool[i * args.docs_per_page:(i + 1) * args.docs_per_page]
                 for i in range(n_pages)]
    stream_docs = pool[n_pages * args.docs_per_page:]

    llm = VLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model)
    maint = (VLLM(host=f"http://127.0.0.1:{args.port}", model=args.maintainer_model)
             if args.maintainer_model else llm)
    t0 = time.time()

    def gen_batch(items, mnt, temp, eng=llm):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(lambda it: eng.gen(it[0], it[1], mnt, temp), items))

    # build initial pages on BASE (compiler)
    pages = gen_batch([(CLEAN_SYS, "\n\n".join(
        f"[Doc {k+1}] {doc_text(corpus_all[d])}" for k, d in enumerate(m)))
        for m in page_docs], 600, 0.7)
    members = [list(m) for m in page_docs]
    print(f"[{args.label}] built {len(pages)} pages; replaying {chain_len} chain steps", flush=True)

    # replay chains deterministically (temp 0, single rewrite) with the chosen maintainer
    for step in range(chain_len):
        new_for_page = {}
        for pi in range(len(pages)):
            idx = step * len(pages) + pi
            if idx < len(stream_docs):
                new_for_page[pi] = stream_docs[idx]
        users = {pi: (f"CURRENT PAGE:\n{pages[pi]}\n\nNEW DOCUMENT:\n"
                      f"{doc_text(corpus_all[d])}") for pi, d in new_for_page.items()}
        outs = gen_batch([(INCR_VANILLA, users[pi]) for pi in users], 600, 0.0, eng=maint)
        for pi, o in zip(users, outs):
            pages[pi] = o
            members[pi].append(new_for_page[pi])
        print(f"  step {step} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # measure retention of P_train / P_held / P_fresh per page (temp-0 judge on BASE)
    def judge_present(page, claim_list):
        items = [(JUDGE_SYS, f"CONTEXT:\n{page}\n\nCLAIM: {cl}") for cl in claim_list]
        outs = gen_batch(items, 4, 0.0)
        return [1 if a.strip().upper().startswith("YES") else 0 for a in outs]

    rows = {"train": [], "held": [], "fresh": []}
    mech = {"train": [], "held": []}
    for pi in range(len(pages)):
        mem = members[pi]
        rw = rewarded_by_page.get(pi, set())
        train_ids = [d for d in mem if d in rw and d in claims]
        held_ids = [d for d in mem if d not in rw and d in claims]
        fresh_ids = [d for d in mem if d in claims2]
        if train_ids:
            ys = judge_present(pages[pi], [claims[d] for d in train_ids])
            rows["train"] += ys
            mech["train"] += [overlap(claims[d], pages[pi]) for d in train_ids]
        if held_ids:
            ys = judge_present(pages[pi], [claims[d] for d in held_ids])
            rows["held"] += ys
            mech["held"] += [overlap(claims[d], pages[pi]) for d in held_ids]
        if fresh_ids:
            rows["fresh"] += judge_present(pages[pi], [claims2[d] for d in fresh_ids])

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    summary = {"label": args.label, "maintainer_model": args.maintainer_model,
               "reward_frac": meta.get("reward_frac"), "word_cap": word_cap,
               "R_train": rate(rows["train"]), "n_train": len(rows["train"]),
               "R_held": rate(rows["held"]), "n_held": len(rows["held"]),
               "R_fresh": rate(rows["fresh"]), "n_fresh": len(rows["fresh"]),
               "train_minus_held": (round(rate(rows["train"]) - rate(rows["held"]), 4)
                                    if rows["train"] and rows["held"] else None),
               "mech_overlap_train": rate(mech["train"]),
               "mech_overlap_held": rate(mech["held"]),
               "minutes": round((time.time() - t0) / 60, 1)}
    json.dump({"summary": summary, "rows": rows, "config": vars(args)},
              open(args.out, "w"), indent=1)
    print(f"=== GOODHART [{args.label}] R_train={summary['R_train']} "
          f"R_held={summary['R_held']} R_fresh={summary['R_fresh']} "
          f"gap={summary['train_minus_held']} ===", flush=True)


if __name__ == "__main__":
    main()
