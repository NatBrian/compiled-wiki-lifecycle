#!/usr/bin/env python3
"""Re-derive EVERY LaTeX results table from the results/ JSONL — no hand-copied numbers.
Emits .tex table bodies into paper/latex/tables/. Run before each LaTeX build.
"""
import json, glob, os
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "latex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def write_rows(path, lines):
    """Write table-body rows; the LAST row omits the trailing '\\\\' because an
    \\input'd file ending in \\\\ before \\bottomrule trips 'Misplaced \\noalign'
    in xetex/tectonic. main.tex adds the final \\\\ after each \\input."""
    body = []
    for i, ln in enumerate(lines):
        ln = ln.rstrip()
        if i == len(lines) - 1 and ln.endswith("\\\\"):
            ln = ln[:-2].rstrip()
        body.append(ln)
    path.write_text("\n".join(body) + "\n")


def stale_acc(fname):
    rows = [json.loads(l) for l in open(RES / fname)]
    n = len(rows)
    x = sum(1 for r in rows if r.get("stale"))
    acc = sum(1 for r in rows if r.get("correct")) / n
    return n, x, acc


def wilson_ci(x, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0, c - h), c + h


def cp_upper(x, n, alpha=0.05):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - alpha, x + 1, n - x))


# ---- Table 1: main SER (14B) ----
MAIN = [
    ("LLM-Wiki (Karpathy)", "overwrite + FTS5 nav", "_ckpt_pool_wiki_karpathy.jsonl"),
    ("Label-free resolver", "recency resolve, \\emph{no labels}", "_ckpt_pool_resolve_free.jsonl"),
    ("Closed-book", "parametric prior", "_ckpt_pool_closed_book.jsonl"),
    ("Full-dump (CiC)", "long-context, no curation", "_ckpt_pool_full_dump_cic.jsonl"),
    ("RAPTOR (real tree-RAG)", "accumulate (recursive summary)", "_ckpt_pool_raptor.jsonl"),
    ("Vector RAG", "dense top-$k$, accumulate", "_ckpt_pool_vector_rag.jsonl"),
    ("LightRAG (real graph-RAG)", "accumulate (concat)", "_ckpt_pool_lightrag.jsonl"),
]
lines = []
for name, mech, f in MAIN:
    n, x, acc = stale_acc(f)
    lo, hi = wilson_ci(x, n)
    bold = "\\textbf" if name.startswith(("LLM-Wiki", "Label-free")) else ""
    ser = f"{x/n:.3f}"
    lines.append(f"{name} & {mech} & {bold}{{{ser}}} & [{lo:.3f}, {hi:.3f}] & {acc:.3f} & {x} \\\\")
write_rows(OUT / "tab_main.tex", lines)

# ---- Table 2: reader-size SER ----
READERS = [
    ("LLM-Wiki", "wiki_karpathy"),
    ("Label-free resolver", "resolve_free"),
    ("Full-dump (CiC)", "full_dump_cic"),
    ("Vector RAG", "vector_rag"),
]
lines = []
for name, tag in READERS:
    cells = []
    for sfx in ("_1p5b", "_7b", ""):
        n, x, _ = stale_acc(f"_ckpt_pool_{tag}{sfx}.jsonl")
        cells.append(f"{x/n:.3f}")
    lines.append(f"{name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
write_rows(OUT / "tab_readers.tex", lines)

# ---- Table 3: LOFT ----
def loft_acc(arm, ds):
    d = json.load(open(RES / f"results_loft_{arm}_{ds}_32k.json"))
    return d["accuracy"], d["n_corpus"]
lines = []
for ds, label in [("nq", "nq (single-hop)"), ("hotpotqa", "hotpotqa (multi-hop)")]:
    fa, nc = loft_acc("full_dump_cic", ds)
    va, _ = loft_acc("vector_rag", ds)
    vb = f"\\textbf{{{va:.3f}}}" if va > fa else f"{va:.3f}"
    lines.append(f"{label} & {nc} & {fa:.3f} & {vb} \\\\")
write_rows(OUT / "tab_loft.tex", lines)

# ---- Table 7: certificate (raw + judge-corrected) ----
ct = {(r["arm"], r["reader"]): r for r in json.loads((RES / "certificate_table.json").read_text())}
corr = {r["arm"]: r for r in json.loads((RES / "certificate_corrected.json").read_text())}
# judge_noise_final: resolver corrected eps from blinded raters
jn = json.loads((RES / "judge_noise_final.json").read_text())
CERT = ["LLM-Wiki (Karpathy)", "Label-free resolver", "Closed-book",
        "Full-dump (CiC)", "RAPTOR (tree-RAG)", "Vector RAG", "LightRAG (graph-RAG)"]
lines = []
for arm in CERT:
    r = ct[(arm, "14B")]
    eps_corr = corr[arm]["eps_corrected_95"]
    if arm == "Label-free resolver":
        eps_corr = jn["corrected_resolver_eps95"]   # 0.080 from blinded-rater FN
    bold = "\\textbf" if arm in ("LLM-Wiki (Karpathy)", "Label-free resolver") else ""
    lines.append(f"{arm} & {r['ser_point']:.3f} & {bold}{{{r['cp_upper_95']:.3f}}} & {eps_corr:.3f} \\\\")
write_rows(OUT / "tab_certificate.tex", lines)

# ---- Table: structure x resolution 2x2 (§4.2d) ----
def ser_of(fname):
    n, x, _ = stale_acc(fname)
    return x / n
g_flat_acc = ser_of("_ckpt_pool_vector_rag.jsonl")
g_flat_res = ser_of("_ckpt_pool_resolve_free.jsonl")
g_graph_acc = ser_of("_ckpt_pool_graph_accumulate.jsonl")
g_graph_res = ser_of("_ckpt_pool_graph_resolve.jsonl")
write_rows(OUT / "tab_dissociation.tex", [
    f"\\textbf{{flat}} (chunks) & Vector RAG\\,---\\,{g_flat_acc:.3f} & label-free resolver\\,---\\,{g_flat_res:.3f} \\\\",
    f"\\textbf{{graph}} (entity nodes) & graph-accumulate\\,---\\,{g_graph_acc:.3f} & \\textbf{{graph-resolve\\,---\\,{g_graph_res:.3f}}} \\\\",
])

# ---- Table: anonymous vs stamped (§4.1) ----
def ser_acc(fname):
    n, x, a = stale_acc(fname)
    return x / n, a
va_s, va_a = ser_acc("_ckpt_pool_vector_rag.jsonl")
vs_s, vs_a = ser_acc("_ckpt_pool_vector_rag_stamped.jsonl")
ga_s, ga_a = ser_acc("_ckpt_pool_graph_accumulate.jsonl")
gs_s, gs_a = ser_acc("_ckpt_pool_graph_accumulate_stamped.jsonl")
write_rows(OUT / "tab_stamped.tex", [
    f"Vector RAG & {va_s:.3f} & \\textbf{{{vs_s:.3f}}} & {va_a:.3f} $\\to$ {vs_a:.3f} \\\\",
    f"Graph (entity) accumulate & {ga_s:.3f} & \\textbf{{{gs_s:.3f}}} & {ga_a:.3f} $\\to$ {gs_a:.3f} \\\\",
])

print("wrote tables:", *(p.name for p in sorted(OUT.glob("*.tex"))))
print("\n-- main SER (sanity) --")
print((OUT / "tab_main.tex").read_text())
print("-- certificate --")
print((OUT / "tab_certificate.tex").read_text())
