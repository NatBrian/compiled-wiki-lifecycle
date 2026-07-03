"""E1b — style-robustness diagnostic for the TPR calibration.

The E1 certificate uses TPR measured on raw-text contexts. It remains a valid
LOWER bound on compiled-store retention iff TPR_raw >= TPR_compiled-style
(judge at least as good at verifying facts from raw text as from wiki prose).

Test: compile each calibration gold doc ALONE into a wiki paragraph (no
compression competition => fact almost surely survives). Judge the claim
against that single-doc page. Observed rate r = TPR_style * survival_1doc
<= TPR_style. If r >= TPR_raw (or close), then TPR_style >= TPR_raw holds
with margin and the certificate direction is safe.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
import data as D
from oai_client import VLLM
from certify import JUDGE_SYS, doc_text, split_claims, judge_many

ONE_SYS = ("You are a wiki compiler. Rewrite the scientific abstract below as a wiki "
           "paragraph (at most 350 words). PRESERVE every key factual finding: specific "
           "entities, quantities, numbers, directions of effect, and comparisons. "
           "Dense declarative prose. No preamble.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "benchmarks", "scifact-open", "data"))
    ap.add_argument("--n_cal", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results_e1b.json"))
    args = ap.parse_args()

    corpus_all = D.load_corpus(os.path.join(args.data, "corpus_candidates.jsonl"))
    claims_all = D.load_claims(os.path.join(args.data, "claims.jsonl"))
    cal_claims, _, _ = split_claims(claims_all, corpus_all, args.n_cal, args.seed)

    llm = VLLM()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pages = list(ex.map(
            lambda c: llm.gen(ONE_SYS, doc_text(corpus_all[c["gold"]]),
                              max_new_tokens=600, temperature=0.7), cal_claims))
    ys = judge_many(llm, list(zip(pages, [c["claim"] for c in cal_claims])),
                    args.workers, "TPR 1-doc-compiled style")
    e1 = json.load(open(os.path.join(HERE, "results_e1.json")))
    tpr_raw_single = e1["report"]["tpr_single"]
    r = sum(ys) / len(ys)
    out = {"tpr_onedoc_compiled_lowerbound": round(r, 4),
           "tpr_raw_single": tpr_raw_single,
           "assumption_safe": bool(r >= tpr_raw_single - 0.05),
           "n": len(ys), "note": "r = TPR_style*survival <= TPR_style; "
           "safe if r >= raw TPR (within noise)"}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
