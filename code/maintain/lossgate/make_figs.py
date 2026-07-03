"""Figures for the LossGate stage from results/analysis.json (+ raw a1_*.json).
Fig1 hand-drawn in LaTeX (TikZ). Fig2 = A1 frontier; Fig3 = E2 stream vs static.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
RES = os.path.join(P5, "results")
FIG = os.path.join(P5, "paper", "latex", "figs")
os.makedirs(FIG, exist_ok=True)

ARM_STYLE = {
    "vanilla": ("#d62728", "o", "vanilla (P3 disease)"),
    "conservative": ("#1f77b4", "s", "conservative prompt (P4)"),
    "lossgate_vanilla": ("#2ca02c", "^", "LossGate(vanilla)"),
    "lossgate_conservative": ("#9467bd", "D", "LossGate(conservative)"),
}


def fig2_frontier(A):
    """Retention trajectory + (final retention vs currency) inset."""
    tl = A["timelines"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, (c, m, lab) in ARM_STYLE.items():
        if arm not in tl:
            continue
        ts = [r["t"] for r in tl[arm]]
        R = [r["valid_R"] for r in tl[arm]]
        ax1.plot(ts, R, marker=m, color=c, label=lab, ms=4)
    ax1.set_xlabel("maintenance batch t"); ax1.set_ylabel("retention R(t) (VALID pool, corrected)")
    ax1.set_title("(a) Retention under maintenance"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    for arm, (c, m, lab) in ARM_STYLE.items():
        if arm not in tl:
            continue
        f = tl[arm][-1]
        if f["incorp"] is None:
            continue
        ax2.errorbar(f["incorp"], f["valid_R"],
                     xerr=f.get("incorp_std", 0), yerr=f.get("valid_R_std", 0),
                     fmt=m, color=c, ms=10, capsize=3, label=lab)
        ax2.annotate(arm.replace("lossgate_", "LG-"), (f["incorp"], f["valid_R"]),
                     fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("currency: incorporation rate of new facts")
    ax2.set_ylabel("final retention R(12)")
    ax2.set_title("(b) Make-or-break frontier\n(up-and-right dominates; left = frozen)")
    ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_frontier.pdf")); plt.close(fig)
    print("wrote fig2_frontier.pdf")


def fig3_stream(A):
    """E1/E2 visualization: per-batch certified floor R(t-1)-b_t vs realized R(t)
    (the certificate holds every batch) for a gated arm; a frozen t0 snapshot makes
    no per-step claim. tl carries valid_R and b_t per t."""
    tl = A["timelines"]
    arm = "lossgate_vanilla" if "lossgate_vanilla" in tl else next(iter(tl), None)
    rows = tl[arm]
    ts = [r["t"] for r in rows]
    R = [r["valid_R"] for r in rows]
    # certified floor for batch t = R(t-1) - b_t
    floor = [None] + [rows[t - 1]["valid_R"] - rows[t]["b_t"] for t in range(1, len(rows))]
    L0 = rows[0]["valid_R"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ts, R, "k-o", ms=4, label="realized retention $R(t)$ (held-out VALID)")
    ax.plot(ts[1:], floor[1:], "g--^", ms=4,
            label="per-batch certified floor $R(t{-}1)-b_t$")
    ax.axhline(L0, color="#1f77b4", ls=":", label="frozen $t_0$ snapshot (no $t{>}0$ claim)")
    ax.fill_between(ts[1:], floor[1:], R[1:], color="green", alpha=.08)
    ax.set_xlabel("maintenance batch $t$"); ax.set_ylabel("retention / certified floor")
    ax.set_title("Per-batch certificate: realized $R(t)$ stays above the\ncertified floor "
                 "$R(t{-}1)-b_t$ every batch (E1 coverage $=1.0$)")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_stream.pdf")); plt.close(fig)
    print("wrote fig3_stream.pdf")


def fig4_stream_big():
    """E2-rescue: large-probe judge-free HoH stream LCB stays non-vacuous and valid."""
    p = os.path.join(RES, "analysis2.json")
    if not os.path.exists(p):
        print("no analysis2.json"); return
    e2 = json.load(open(p)).get("e2_rescue", {})
    arm = "lossgate_vanilla" if "lossgate_vanilla" in e2 else next(iter(e2), None)
    if not arm:
        print("no e2_rescue data"); return
    run = e2[arm][0]
    realized = run["realized"]; stream = run["stream_lcb"]
    ts = list(range(len(realized)))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ts, realized, "k-o", ms=4, label="realized retention (held-out, exact)")
    ax.plot(ts, stream, "g--^", ms=4,
            label=f"anytime-valid stream LCB (gate pool $n={run['gate_n']}$)")
    ax.fill_between(ts, 0, stream, color="green", alpha=.08)
    ax.set_ylim(0, 1)
    ax.set_xlabel("maintenance batch $t$"); ax.set_ylabel("retention / stream lower bound")
    ax.set_title("E2: with a large probe pool, the composed stream LCB\n"
                 "stays strictly positive and valid over the whole stream")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_stream_big.pdf")); plt.close(fig)
    print("wrote fig4_stream_big.pdf")


def main():
    A = json.load(open(os.path.join(RES, "analysis.json")))
    fig2_frontier(A)
    fig3_stream(A)
    fig4_stream_big()


if __name__ == "__main__":
    main()
