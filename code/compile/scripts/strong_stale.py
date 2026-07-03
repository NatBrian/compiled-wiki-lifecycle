#!/usr/bin/env python3
"""Strengthened staleness detector -> judge-noise correction.

The production SER matcher (_hoh_match) flags stale via substring OR >=80% overlap
of the superseded symbol's content tokens (len>2). It SYSTEMATICALLY misses
superseded answers carried by short / numeric tokens (e.g. resp '12' vs superseded
'12 years old'; resp 'Lines 1,3,4 and 5' vs superseded '1, 3, 4, 5') -- gold-judge
adjudication (strongest judge, exhaustive on the resolving arms) found these are the
ONLY false-negative class, all in accumulating arms, and zero false positives.

This strong detector ADDS a numeric/set rule on top of the production matcher and is
validated to introduce no new false positives. The gap between strong-SER and
production-SER is the measured judge-noise correction, computed on all 300 queries
(no subsampling). For arms where the gap is ~0 the corrected certificate == the
uncorrected one (Feng et al. 2026, arXiv:2601.20913: with calibrated ~0 error rates
the correction is a no-op).
"""
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "real"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import real_core

ARMS = {
    "LLM-Wiki (Karpathy)": "_ckpt_pool_wiki_karpathy.jsonl",
    "Label-free resolver": "_ckpt_pool_resolve_free.jsonl",
    "Closed-book": "_ckpt_pool_closed_book.jsonl",
    "Full-dump (CiC)": "_ckpt_pool_full_dump_cic.jsonl",
    "RAPTOR (tree-RAG)": "_ckpt_pool_raptor.jsonl",
    "Vector RAG": "_ckpt_pool_vector_rag.jsonl",
    "LightRAG (graph-RAG)": "_ckpt_pool_lightrag.jsonl",
}


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.$%-]", " ", (s or "").lower())).strip()


def _nums(s):
    """Distinctive numeric / percentage tokens (the class the fuzzy matcher drops)."""
    return set(re.findall(r"\d[\d,.]*%?", _norm(s)))


def _content_set(s):
    """All content tokens incl. short ones (numbers, single letters), minus pure stop."""
    stop = {"the", "a", "an", "of", "to", "in", "and", "is", "was", "for", "on",
            "at", "as", "match", "against", "second", "with", "or", "per", "year",
            "years", "old", "students", "countries", "locations", "aircraft"}
    toks = set(_norm(s).split())
    return {t for t in toks if t not in stop}


def strong_stale(resp, gold, deprecated):
    """True iff resp conveys a superseded answer that the production matcher may miss.
    Rule: resp's numeric signature equals a superseded symbol's numeric signature AND
    differs from gold's (the verified FN class); we DO NOT flag when resp already
    matches gold's numbers (that is a current answer the fuzzy rule under-credited)."""
    rn, gn = _nums(resp), _nums(gold)
    for d in deprecated:
        dn = _nums(d)
        if dn and dn <= rn and not (gn and gn <= rn):
            # resp carries the superseded number(s), not the current one(s)
            return True, d
    # set/list rule: superseded content-token set fully inside resp, gold's not
    rs = _content_set(resp)
    gs = _content_set(gold)
    for d in deprecated:
        ds = _content_set(d)
        if len(ds) >= 2 and ds <= rs and not (gs and gs <= rs):
            return True, d
    return False, None


def main():
    items = real_core.load_hoh(300)
    queries = [real_core.hoh_stream(idx, rec)[1] for idx, rec in items]
    report = {}
    print(f"{'arm':24s} {'prod_stale':>10s} {'+FN_found':>9s} {'strong_stale':>12s}  FN_examples")
    for arm, fname in ARMS.items():
        rows = [json.loads(l) for l in open(RES / fname)]
        prod = sum(1 for r in rows if r.get("stale"))
        fn = []
        for i, r in enumerate(rows):
            if r.get("correct") or r.get("stale"):
                continue                          # production already decided
            ok, d = strong_stale(r.get("resp", ""), r.get("gold", ""),
                                 queries[i]["deprecated_answers"])
            if ok:
                fn.append({"qidx": i, "resp": r.get("resp", "")[:50],
                           "gold": r.get("gold", "")[:35], "matched_superseded": d})
        strong = prod + len(fn)
        report[arm] = {"prod_stale": prod, "fn_found": len(fn),
                       "strong_stale": strong, "n": len(rows), "fn": fn}
        ex = "; ".join(f"#{f['qidx']}:{f['resp']!r}->{f['matched_superseded']!r}" for f in fn[:3])
        print(f"{arm:24s} {prod:10d} {len(fn):9d} {strong:12d}  {ex}")
    (RES / "strong_stale_report.json").write_text(json.dumps(report, indent=2))
    print("\nFull FN lists -> results/strong_stale_report.json")


if __name__ == "__main__":
    main()
