"""Trained-maintainer stage: maintenance-arm protocol (constant-corpus
rewrite-collapse harness).

Reuses the certify stage's machinery (compile, calibrated judge, embedding router, LCB),
carried through this stage's earlier diagnosis work, and adds ONLY arm
selection: a maintainer = (rewrite PROMPT, serving MODEL). Build + judge + clean-rebuild
always run on the BASE model; only the per-document rewrite uses the arm's prompt and
(optionally) a LoRA adapter. This isolates "does the maintainer preserve facts" from
compiler/judge behaviour, exactly as this stage's earlier diagnosis work did in its E0/E6.

Constant-corpus regime (--replace_fillers, default ON): each batch swaps `batch_fillers`
filler docs out and `batch_fillers` new ones in, holding corpus size fixed, so any retention
change is iterative-rewrite damage rather than page-budget crowding.

Arms (--arm): vanilla | conservative | anchored   (+ trained via --maintainer_model)

Output JSON schema is a strict superset of e0_pilot.py's so this stage's decay-law-fitting
analysis (p4_analysis.py) works unchanged. Every probe verdict is judge-noise corrected
with the SAME calibration as the certify stage and this stage's earlier diagnosis work
(contract/results_e1.json) unless --cal_file overrides.

Run (detached): ./detached_run.sh a1_cons_s0 results/results_a1_conservative_seed0.json -- \
  python p4_protocol.py --arm conservative --seed 0 \
    --out results/results_a1_conservative_seed0.json
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
from seeded_client import SeededVLLM
from certify import CLEAN_SYS, doc_text, split_claims, assign_pages, judge_many
from maintain import Store, PageRouter
from stats import retention_lcb, corrected_point

WORD_CAP = 350

# ---- maintainer rewrite prompts (the arm knob) ---------------------------------------
VANILLA = (  # this stage's earlier diagnosis work's INCR_SYS — the disease baseline (verbatim)
    f"You maintain a wiki page. Rewrite the page to incorporate the key factual "
    f"findings of the NEW document (entities, quantities, directions of effect, "
    f"comparisons) while keeping the important facts already on the page. "
    f"The rewritten page must stay under {WORD_CAP} words. Output ONLY the new page text.")

CONSERVATIVE = (  # the decisive cheap baseline: minimal edit, never drop
    f"You maintain a wiki page. A NEW document has arrived. Make the SMALLEST possible "
    f"change to the page so it also captures the new document's key factual findings. "
    f"Rules: (1) APPEND the new findings; do NOT rewrite, re-summarize, or rephrase "
    f"sentences that are already on the page. (2) NEVER delete or shorten an existing "
    f"factual statement unless the new document DIRECTLY CONTRADICTS it — then replace "
    f"only the contradicted fact and keep everything else. (3) Keep the page under "
    f"{WORD_CAP} words; only if you would exceed it, compress the single least-specific "
    f"sentence. Output ONLY the new page text.")

ANCHORED = (  # strongest prompt: treat the page as a preserved claim list
    f"You maintain a wiki page that records distinct factual claims. Treat every existing "
    f"claim as ANCHORED: reproduce it verbatim. Add the NEW document's key findings as "
    f"additional claims. Only edit an existing claim if the new document directly "
    f"contradicts it, changing only that one claim. Do not paraphrase, merge, summarize, "
    f"or drop existing claims. Stay under {WORD_CAP} words; if full, drop only the least "
    f"specific NEW claim rather than any existing one. Output ONLY the new page text.")

PROMPTS = {"vanilla": VANILLA, "conservative": CONSERVATIVE, "anchored": ANCHORED}

# incorporation check: did a NEW document's key fact actually make it onto its host page?
# Guards against the "lazy maintainer" failure (preserve old facts by refusing to absorb new
# ones). A genuine maintainer should incorporate at a rate comparable to vanilla.
INCORP_CLAIM_SYS = ("State the single most important specific factual finding of this scientific "
                    "abstract in ONE self-contained sentence (name the entities and the direction "
                    "or size of the effect). Output only that sentence.")
JUDGE_SYS_LOCAL = ("You verify claims against a context. Answer YES only if the context contains "
                   "information supporting the claim (paraphrase counts; the specific finding must "
                   "be present). Otherwise answer NO. Reply with exactly one word: YES or NO.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(PROMPTS), default="vanilla",
                    help="maintainer rewrite prompt")
    ap.add_argument("--maintainer_model", default=None,
                    help="route ONLY rewrites to this served model (e.g. a LoRA adapter "
                         "name); build/judge/clean stay on --base_model")
    ap.add_argument("--base_model", default="qwen2.5-14b")
    ap.add_argument("--maint_port", type=int, default=None,
                    help="serve the maintainer model from a DIFFERENT vLLM (e.g. a larger/smaller "
                         "model on another GPU) while build/judge stay on --port; for scale checks")
    ap.add_argument("--data",  # external dataset, not bundled in this repo -- fetch SciFact-Open
                    default=os.path.join(ROOT, "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_docs0", type=int, default=200)
    ap.add_argument("--docs_per_page", type=int, default=8)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_fillers", type=int, default=40)
    ap.add_argument("--replace_fillers", type=int, default=1,
                    help="1=constant-corpus (isolate rewrite damage); 0=growing corpus")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[0, 12],
                    help="batches at which to compute clean-rebuild reference band")
    ap.add_argument("--n_clean", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--samp_seed", type=int, default=None,
                    help="vLLM sampling seed injected into every request (default: --seed). "
                         "Decouples generation draws from corpus shuffle so distinct seeds are "
                         "independent sampling replications (variance-honesty fix).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")))
    ap.add_argument("--cal_file",
                    default=os.path.join(P2, "results_e1.json"))
    ap.add_argument("--save_pages", action="store_true",
                    help="store full page texts each batch (for mechanism diffing)")
    ap.add_argument("--track_incorp", type=int, default=0,
                    help="if >0, each batch sample this many newly-ingested docs and check whether "
                         "their key fact landed on the host page (incorporation rate)")
    ap.add_argument("--max_incr", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    incr_sys = PROMPTS[args.arm]

    # ---- data + probe set (FIXED at split seed 0, matches the certify stage's and
    #      this stage's earlier calibration) ----------
    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    _, audit_claims, _ = split_claims(claims_all, corpus_all, 40, 0)
    seen = set()
    probes = [c for c in audit_claims
              if not (c["claim"] in seen or seen.add(c["claim"]))]
    gold0 = {g for c in probes for g in c["all_golds"]}
    rng = random.Random(args.seed)
    fillers = [i for i in corpus_all if i not in gold0]
    rng.shuffle(fillers)
    n_fill0 = max(0, args.n_docs0 - len(gold0))
    corpus0 = {i: corpus_all[i] for i in list(gold0) + fillers[:n_fill0]}
    stream = fillers[n_fill0:]
    print(f"arm={args.arm} maint_model={args.maintainer_model} seed={args.seed} "
          f"probes={len(probes)} corpus0={len(corpus0)} stream={len(stream)}", flush=True)

    # ---- judge calibration (shared with the certify stage and this stage's earlier
    #      diagnosis work) ----------------------------------------
    e1 = json.load(open(args.cal_file))
    vv = e1["verdicts"]
    cal = {"k_tpr": sum(vv["tpr_single"]), "n_tpr": len(vv["tpr_single"]),
           "k_fpr": sum(vv["fpr_wiki"]), "n_fpr": len(vv["fpr_wiki"])}
    tpr, fpr = cal["k_tpr"] / cal["n_tpr"], cal["k_fpr"] / cal["n_fpr"]

    def metrics(v):
        k, n = sum(v.values()), len(v)
        lcb, _ = retention_lcb(k, n, cal["k_tpr"], cal["n_tpr"],
                               cal["k_fpr"], cal["n_fpr"], alpha=args.alpha)
        return {"raw": round(k / n, 4),
                "corrected": round(corrected_point(k / n, tpr, fpr), 4),
                "lcb": round(lcb, 4)}

    # ---- servers: base for build/judge/clean; (maybe) adapter for rewrites ------------
    samp_seed = args.seed if args.samp_seed is None else args.samp_seed
    llm = SeededVLLM(host=f"http://127.0.0.1:{args.port}", model=args.base_model, seed=samp_seed)
    maint_host = f"http://127.0.0.1:{args.maint_port or args.port}"
    maint_llm = (SeededVLLM(host=maint_host, model=args.maintainer_model, seed=samp_seed)
                 if args.maintainer_model else llm)

    t0 = time.time()
    gen_calls = {"build": 0, "maint": 0}

    # build store on BASE model (Store uses CLEAN_SYS @ temp 0.7)
    A = Store(llm, corpus0, args.docs_per_page, args.seed, args.workers, "A")
    gen_calls["build"] += len(A.pages)
    router = PageRouter()
    rewrites = [0] * 4096

    def judge_store(pages, page_of, tag):
        pairs = [(pages[page_of[c["gold"]]], c["claim"]) for c in probes]
        ys = judge_many(llm, pairs, args.workers, tag)
        return {c["claim"]: y for c, y in zip(probes, ys)}

    def ingest(new_docs):
        """Route each new doc to nearest page; rewrite with the arm's prompt+model."""
        changed = set()
        for did, doc in new_docs.items():
            pi = router.nearest(doc_text(doc), A.pages)
            user = f"CURRENT PAGE:\n{A.pages[pi]}\n\nNEW DOCUMENT:\n{doc_text(doc)}"
            A.pages[pi] = maint_llm.gen(incr_sys, user,
                                        max_new_tokens=args.max_incr, temperature=0.7)
            gen_calls["maint"] += 1
            A.corpus[did] = doc
            A.members[pi].append(did)
            A.page_of_doc[did] = pi
            rewrites[pi] += 1
            changed.add(pi)
        return changed

    def clean_band(t):
        """Fresh one-shot compiles of A's current corpus snapshot (the honest-rebuild
        reference), at both fixed docs/page and A's current page budget."""
        snap = dict(A.corpus)
        per_budget = max(1, -(-len(snap) // len(A.pages)))
        out = {"t": t, "n_docs": len(snap), "clean": {}}
        for flavor, pp in (("std", args.docs_per_page), ("budget", per_budget)):
            builds = []
            for b in range(args.n_clean):
                members = assign_pages(snap.keys(), pp, args.seed * 100 + t * 10 + b)
                items = [(CLEAN_SYS, "\n\n".join(
                    f"[Doc {k+1}] {doc_text(snap[i])}" for k, i in enumerate(m)))
                    for m in members]
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    pages = list(ex.map(lambda it: llm.gen(it[0], it[1], 600, 0.7), items))
                page_of = {d: pi for pi, m in enumerate(members) for d in m}
                v = judge_store(pages, page_of, f"clean t{t} {flavor}#{b}")
                builds.append({"metrics": metrics(v), "n_pages": len(pages)})
            out["clean"][flavor] = {
                "builds": builds,
                "corrected_mean": round(
                    sum(x["metrics"]["corrected"] for x in builds) / len(builds), 4)}
        return out

    # ---- t=0 ----
    vA = judge_store(A.pages, A.page_of_doc, "t0")
    timeline = [{"t": 0, "verdicts": vA, "metrics": metrics(vA), "n_changed": 0,
                 "page_words": [len(p.split()) for p in A.pages],
                 "rewrites": rewrites[:len(A.pages)],
                 "store_words": sum(len(p.split()) for p in A.pages),
                 "gen_calls": dict(gen_calls),
                 **({"pages": list(A.pages)} if args.save_pages else {})}]
    checkpoints = []
    if 0 in args.checkpoints:
        checkpoints.append(clean_band(0))

    def incorp_rate(new_doc_ids):
        """Fraction of a sample of just-ingested docs whose key fact is now on the host page."""
        sample = new_doc_ids[:args.track_incorp]
        if not sample:
            return None
        claims = [llm.gen(INCORP_CLAIM_SYS, doc_text(A.corpus[d]), 80, 0.0).strip()
                  for d in sample]
        pairs = [(A.pages[A.page_of_doc[d]], cl) for d, cl in zip(sample, claims)]
        ys = judge_many(llm, pairs, args.workers, "incorp")
        return round(sum(ys) / len(ys), 4)

    # ---- maintenance stream ----
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
        changed = ingest(new_docs)
        ir = incorp_rate(list(new_docs)) if args.track_incorp else None
        vA = judge_store(A.pages, A.page_of_doc, f"t{t}")
        m = metrics(vA)
        timeline.append({"t": t, "verdicts": vA, "metrics": m, "n_changed": len(changed),
                         "incorp_rate": ir,
                         "page_words": [len(p.split()) for p in A.pages],
                         "rewrites": rewrites[:len(A.pages)],
                         "store_words": sum(len(p.split()) for p in A.pages),
                         "gen_calls": dict(gen_calls),
                         **({"pages": list(A.pages)} if args.save_pages else {})})
        print(f"[{args.arm}] t={t}: changed={len(changed)} raw={m['raw']} "
              f"corrected={m['corrected']} lcb={m['lcb']}", flush=True)
        if t in args.checkpoints:
            checkpoints.append(clean_band(t))

    json.dump({"config": vars(args), "cal": cal,
               "page_of_probe": {c["claim"]: A.page_of_doc[c["gold"]] for c in probes},
               "probes": [{"claim": c["claim"], "gold": c["gold"]} for c in probes],
               "timeline": timeline, "checkpoints": checkpoints,
               "gen_calls": gen_calls, "n_calls": llm.n_calls + maint_llm.n_calls,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(args.out, "w"), indent=1)
    print(f"=== P4 {args.arm} seed {args.seed} DONE {(time.time()-t0)/60:.1f} min "
          f"build={gen_calls['build']} maint={gen_calls['maint']} ===", flush=True)


if __name__ == "__main__":
    main()
