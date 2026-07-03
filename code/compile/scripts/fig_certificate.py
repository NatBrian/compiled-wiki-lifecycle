#!/usr/bin/env python3
"""Figures for the Certified Currency section.

  fig_samplesize.pdf/png : eps_hat (95% CP upper bound on SER) vs audit size n, for a
                           store at wiki-level true SER (~0.007) and for a clean audit
                           (x=0). Shows certification is cheap.
  fig_certificate.pdf/png: per-arm point SER + 95% CP upper bound (the certificate),
                           resolving vs accumulating, 14B reader.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- sample-size curve ----
sc = json.loads((RES / "samplesize_curve.json").read_text())
ns = [c["n"] for c in sc["observed_rate"]]
eps_obs = [c["eps_hat_95"] for c in sc["observed_rate"]]
eps_zero = [z["eps_hat_95"] for z in sc["zero_stale"]]

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(ns, eps_obs, "o-", color="#1f77b4", lw=1.8, ms=4,
        label=r"audit observes wiki-level SER ($\approx$0.007)")
ax.plot(ns, eps_zero, "s--", color="#2ca02c", lw=1.6, ms=4,
        label=r"clean audit (0 stale found)")
ax.axhline(0.05, color="grey", ls=":", lw=1)
ax.text(ns[-1], 0.052, r"$\epsilon$=0.05", ha="right", va="bottom", fontsize=8, color="grey")
ax.set_xscale("log")
ax.set_xlabel("audit size $n$ (queries)")
ax.set_ylabel(r"certified $\hat{\epsilon}$  (95% upper bound on SER)")
ax.set_title("Cost of a currency certificate", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3, which="both")
ax.set_ylim(0, 0.065)
fig.tight_layout()
fig.savefig(FIG / "fig_samplesize.pdf"); fig.savefig(FIG / "fig_samplesize.png", dpi=150)
print("wrote", FIG / "fig_samplesize.pdf")

# ---- certificate forest plot ----
ct = json.loads((RES / "certificate_table.json").read_text())
rows = [r for r in ct if r["reader"] == "14B"]
order = ["LLM-Wiki (Karpathy)", "Label-free resolver", "Closed-book",
         "RAPTOR (tree-RAG)", "Full-dump (CiC)", "Vector RAG", "LightRAG (graph-RAG)"]
rows = sorted(rows, key=lambda r: order.index(r["arm"]))
ys = list(range(len(rows)))[::-1]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for y, r in zip(ys, rows):
    col = "#2ca02c" if r["resolving"] else ("#7f7f7f" if r["arm"] == "Closed-book" else "#d62728")
    ax.plot([r["ser_point"], r["cp_upper_95"]], [y, y], color=col, lw=2.2, alpha=0.85)
    ax.plot(r["ser_point"], y, "o", color=col, ms=5)
    ax.plot(r["cp_upper_95"], y, "|", color=col, ms=10, mew=2)
    ax.text(r["cp_upper_95"] + 0.006, y, rf"$\hat\epsilon$={r['cp_upper_95']:.3f}",
            va="center", fontsize=7.5, color=col)
ax.set_yticks(ys)
ax.set_yticklabels([r["arm"] for r in rows], fontsize=8)
ax.set_xlabel("Staleness Error Rate  (point  •——|  95% certified upper bound)")
ax.set_title("Per-store currency certificate (HoH, n=300, 14B)", fontsize=10)
ax.axvline(0.05, color="grey", ls=":", lw=1)
ax.set_xlim(-0.01, 0.42)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_certificate.pdf"); fig.savefig(FIG / "fig_certificate.png", dpi=150)
print("wrote", FIG / "fig_certificate.pdf")
