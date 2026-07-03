"""Generate paper figures from results JSON. Outputs to paper/latex/figs/."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "latex", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def fig_currency():
    r = json.load(open(os.path.join(HERE, "results_e5.json")))
    tl = r["timeline"]
    ts = [e["t"] for e in tl]
    ret = [tl[0]["retention_LCB"]] + [e["maintained"]["retention_LCB"] for e in tl[1:]]
    ser = [tl[0]["SER_UB"]] + [e["maintained"]["SER_UB"] for e in tl[1:]]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ts, ret, "o-", color="#1b7837", label="fidelity: retention LCB")
    ax.plot(ts, ser, "s--", color="#b2182b", label="currency: SER upper bound")
    ax.axhline(0.8, color="#1b7837", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0.05, color="#b2182b", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("update batch (supersession events absorbed)")
    ax.set_ylabel("certified bound")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Maintained two-clause contract on a supersession stream")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    ax.annotate("born stale", (0, 0.75), (0.4, 0.86), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "currency.pdf"))
    print("wrote currency.pdf")


def fig_cost():
    r = json.load(open(os.path.join(HERE, "results_e2e3.json")))
    tl = r["timeline"]
    ts = [e["t"] for e in tl]
    full = [e["cost"]["full"] for e in tl]
    maint = [e["cost"]["maintained"] for e in tl]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ts, full, "o-", color="#404040", label="full re-audit")
    ax.plot(ts, maint, "s-", color="#2166ac", label="maintained (ours)")
    ax.fill_between(ts, maint, full, color="#2166ac", alpha=0.12)
    ax.set_xlabel("update batch")
    ax.set_ylabel("cumulative audit calls")
    ax.set_title("Maintenance cost: maintained vs full re-audit")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.annotate(f"{maint[-1]}/{full[-1]} = {maint[-1]/full[-1]*100:.0f}%",
                (ts[-1], maint[-1]), (ts[-1]-2.5, maint[-1]+90), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost.pdf"))
    print("wrote cost.pdf")


if __name__ == "__main__":
    fig_currency()
    fig_cost()
