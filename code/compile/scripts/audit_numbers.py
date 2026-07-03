#!/usr/bin/env python3
"""Number audit: assert every headline numeric claim in the abstract, the main SER
table, and the certificate table re-derives EXACTLY from the results/ JSONL. Fails
loudly on any mismatch. Run as the final QA gate before submission.

  python3 paper/scripts/audit_numbers.py    # exit 0 = all claims verified
"""
import json, sys
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
fails = []


def chk(label, got, want, tol=0.0005):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'XX '}] {label}: derived={got:.4f} claimed={want:.4f}")
    if not ok:
        fails.append(label)


def ser(fname):
    rows = [json.loads(l) for l in open(RES / fname)]
    n = len(rows)
    x = sum(1 for r in rows if r.get("stale"))
    acc = sum(1 for r in rows if r.get("correct")) / n
    return x / n, acc, x, n


def cp_upper(x, n, a=0.05):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - a, x + 1, n - x))


print("== Abstract & Table 1: SER and accuracy (14B) ==")
claims = {  # (file, claimed_ser, claimed_acc) -- claimed values are what the paper prints
    "_ckpt_pool_wiki_karpathy.jsonl": (0.007, 0.910),
    "_ckpt_pool_resolve_free.jsonl": (0.050, 0.673),
    "_ckpt_pool_closed_book.jsonl": (0.043, 0.060),
    "_ckpt_pool_full_dump_cic.jsonl": (0.143, 0.607),
    "_ckpt_pool_raptor.jsonl": (0.177, 0.537),
    "_ckpt_pool_vector_rag.jsonl": (0.277, 0.493),
    "_ckpt_pool_lightrag.jsonl": (0.280, 0.527),
}
for f, (cser, cacc) in claims.items():
    s, a, x, n = ser(f)
    chk(f"{f} SER", s, cser)
    chk(f"{f} acc", a, cacc)

print("\n== Table 2: reader-size SER (small-reader claims) ==")
chk("wiki 1.5B SER", ser("_ckpt_pool_wiki_karpathy_1p5b.jsonl")[0], 0.000)
chk("resolver 1.5B SER", ser("_ckpt_pool_resolve_free_1p5b.jsonl")[0], 0.080)
chk("vector 1.5B SER", ser("_ckpt_pool_vector_rag_1p5b.jsonl")[0], 0.390)
chk("dump 1.5B SER", ser("_ckpt_pool_full_dump_cic_1p5b.jsonl")[0], 0.307)

print("\n== Table 5: certificate Clopper-Pearson 95% upper bounds ==")
cert_claims = {  # arm file -> claimed eps_hat (raw)
    "_ckpt_pool_wiki_karpathy.jsonl": 0.021,
    "_ckpt_pool_resolve_free.jsonl": 0.076,
    "_ckpt_pool_full_dump_cic.jsonl": 0.181,
    "_ckpt_pool_raptor.jsonl": 0.217,
    "_ckpt_pool_vector_rag.jsonl": 0.322,
    "_ckpt_pool_lightrag.jsonl": 0.326,
}
for f, ceps in cert_claims.items():
    _, _, x, n = ser(f)
    chk(f"{f} eps_hat@95", cp_upper(x, n), ceps)

print("\n== Abstract claim: resolving eps <= 0.08, accumulating eps >= 0.18 ==")
res_max = max(cp_upper(*ser(f)[2:]) for f in
              ["_ckpt_pool_wiki_karpathy.jsonl", "_ckpt_pool_resolve_free.jsonl"])
acc_min = min(cp_upper(*ser(f)[2:]) for f in
              ["_ckpt_pool_full_dump_cic.jsonl", "_ckpt_pool_raptor.jsonl",
               "_ckpt_pool_vector_rag.jsonl", "_ckpt_pool_lightrag.jsonl"])
print(f"  resolving max eps={res_max:.4f} (claim <=0.08): {'OK' if res_max <= 0.08 else 'XX'}")
print(f"  accumulating min eps={acc_min:.4f} (claim >=0.18): {'OK' if acc_min >= 0.18 else 'XX'}")
if res_max > 0.08:
    fails.append("resolving<=0.08")
if acc_min < 0.18:
    fails.append("accumulating>=0.18")

print("\n== §4.2d structure x resolution 2x2 + §4.1 stamped ==")
chk("graph_accumulate SER", ser("_ckpt_pool_graph_accumulate.jsonl")[0], 0.137)
chk("graph_resolve SER", ser("_ckpt_pool_graph_resolve.jsonl")[0], 0.003)
chk("vector_rag stamped SER", ser("_ckpt_pool_vector_rag_stamped.jsonl")[0], 0.017)
chk("graph_accumulate stamped SER", ser("_ckpt_pool_graph_accumulate_stamped.jsonl")[0], 0.003)

print("\n== 2nd-substrate certificate (Pydantic code library) ==")
_pyd = json.loads((RES / "results_anon_pyd_14b.json").read_text())["arms"]
def _codelib_eps(armkey):
    rs = [r for r in _pyd[armkey]["rows"] if str(r.get("type", "")).startswith("current-fact")]
    n = len(rs); x = sum(1 for r in rs if r.get("stale"))
    return cp_upper(x, n)
chk("Pydantic Vector RAG eps95", _codelib_eps("A1_vector_rag"), 0.133)
chk("Pydantic GraphRAG eps95", _codelib_eps("A2_graph_rag"), 0.034)
chk("Pydantic LLM-Wiki eps95", _codelib_eps("A4_llm_wiki"), 0.034)

print("\n== Judge-noise final (de-circularized) cross-check vs JSON ==")
jn = json.loads((RES / "judge_noise_final.json").read_text())
chk("matcher FPR consensus", jn["matcher_fp_consensus"] / jn["matcher_fp_denom"], 0.05)
chk("resolver corrected eps", jn["corrected_resolver_eps95"], 0.080)
print(f"  kappa={jn['cohens_kappa']}, resolving FN={jn['resolving_fn_consensus']}/{jn['resolving_fn_denom']}")

print("\n" + ("AUDIT PASSED: all claims re-derive from results/." if not fails
              else f"AUDIT FAILED on {len(fails)}: {fails}"))
sys.exit(1 if fails else 0)
