"""Trained-maintainer stage figures. Reads results/analysis_*.json (+ raw
hohmc/goodhart results) and writes
paper/figs/*.pdf. Every figure is robust to missing inputs (skips with a message), so it can
be run incrementally as experiments land.

  fig_decay   : R(t) per arm (SciFact) + fresh-rebuild band  [analysis_main.json]
  fig_cost    : fact half-life (or R(T)) vs deployment cost frontier  [analysis_main.json]
  fig_goodhart: P_train/P_held/P_fresh bars (vanilla vs trained) + dose-response  [goodhart_*.json]
  fig_hoh     : judge-free retention(t) per arm (HoH)  [analysis_hoh.json]
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # bundled repo: no nested code/, scripts live here
RES = os.path.join(P4, "results")
FIGS = os.path.join(P4, "paper", "latex", "figs")
os.makedirs(FIGS, exist_ok=True)

NICE = {"a1_vanilla": "vanilla", "vanilla": "vanilla", "a1_conservative": "conservative",
        "conservative": "conservative", "a1_anchored": "anchored", "anchored": "anchored",
        "b1_trained": "trained (SFT)", "trained": "trained (SFT)",
        "a1_trainedproto": "trained (proto)", "trainedproto_incorp": "trained (proto)",
        "b1_vanillaincorp": "vanilla", "inc_conservative": "conservative",
        "inc_anchored": "anchored", "ledger": "ledger", "rebuild": "rebuild", "union": "union"}
ORDER = ["vanilla", "conservative", "anchored", "ledger", "rebuild", "union", "trained (SFT)"]


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _placeholder(stem, msg):
    """Write a 'pending' placeholder PDF so the draft compiles before data lands."""
    out = os.path.join(FIGS, stem + ".pdf")
    if os.path.exists(out):
        return
    plt.figure(figsize=(4, 2.5)); plt.axis("off")
    plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
    plt.savefig(out); plt.close()


def fig_decay(analysis="analysis_main.json"):
    a = _load(analysis)
    if not a:
        print(f"[fig_decay] missing {analysis}"); return
    plt.figure(figsize=(5.4, 3.6))
    arms = a["arms"]
    band = None
    for tag, d in arms.items():
        nm = NICE.get(tag, tag)
        ts = list(range(len(d["mean_R"])))
        plt.plot(ts, d["mean_R"], marker="o", ms=3, label=nm)
        if d.get("clean_band"):
            band = max(band or 0, max(d["clean_band"].values()))
    clean_ref = a.get("tournament", {}).get("clean_ref_RT")
    if clean_ref:
        plt.axhline(clean_ref, ls="--", color="gray", lw=1.2,
                    label=f"fresh rebuild ({clean_ref:.2f})")
    plt.xlabel("maintenance batch $t$"); plt.ylabel("corrected retention $R(t)$")
    plt.ylim(0, 0.8); plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_decay.pdf")); plt.close()
    print("[fig_decay] ok")


def fig_cost(analysis="analysis_main.json"):
    a = _load(analysis)
    if not a:
        print(f"[fig_cost] missing {analysis}"); return
    tab = a.get("tournament", {}).get("table", [])
    if not tab:
        print("[fig_cost] no tournament table"); return
    plt.figure(figsize=(5.0, 3.6))
    for r in tab:
        nm = NICE.get(r["arm"], r["arm"])
        x = r["deploy_calls"] or 0
        y = r["RT"]
        plt.scatter(x, y, s=40)
        plt.annotate(nm, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    cr = a["tournament"].get("clean_ref_RT")
    if cr:
        plt.axhline(cr, ls="--", color="gray", lw=1, label=f"fresh rebuild ({cr:.2f})")
    plt.xlabel("deployment cost (maintainer calls / run)")
    plt.ylabel("retention $R(12)$"); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_cost.pdf")); plt.close()
    print("[fig_cost] ok")


def fig_frontier(analysis="analysis_v2.json"):
    """Preservation (R12) vs incorporation from ONE matched run-set, with x/y error bars
    (sd across seeds). Replaces the cross-run-set frontier the reviewers flagged."""
    a = _load(analysis) or _load("analysis_frontier.json")
    if not a:
        _placeholder("fig_frontier", "frontier pending"); print("[fig_frontier] placeholder"); return
    tab = [r for r in a.get("tournament", {}).get("table", []) if r.get("incorp_rate") is not None]
    if not tab:
        _placeholder("fig_frontier", "frontier pending"); print("[fig_frontier] no incorp data"); return
    plt.figure(figsize=(5.2, 3.9))
    for r in tab:
        nm = NICE.get(r["arm"], r["arm"])
        x, y = r["incorp_rate"], r["RT"]
        trained = "train" in r["arm"] or "proto" in r["arm"]
        xe = r.get("incorp_sd") or 0
        ye = r.get("sd_RT") or 0
        plt.errorbar(x, y, xerr=xe, yerr=ye, fmt=("*" if trained else "o"),
                     ms=11 if trained else 7, c=("crimson" if trained else "steelblue"),
                     ecolor="gray", elinewidth=0.8, capsize=2, zorder=3)
        plt.annotate(nm, (x, y), fontsize=7, xytext=(6, 3), textcoords="offset points")
    cr = a["tournament"].get("clean_ref_RT")
    if cr:
        plt.axhline(cr, ls="--", color="gray", lw=1, label=f"fresh rebuild ({cr:.2f})")
    plt.xlabel("incorporation rate (new facts absorbed)")
    plt.ylabel("preservation $R(12)$"); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_frontier.pdf")); plt.close()
    print("[fig_frontier] ok")


def fig_trap():
    """The budget-governed adaptivity dissociation: trained P_train-P_held gap vs page budget,
    for vanilla (control) vs trained, at two reward fractions. Reads goodhart_*.json."""
    import numpy as np
    files = sorted(glob.glob(os.path.join(RES, "goodhart_*.json")))
    recs = []
    for fp in files:
        s = json.load(open(fp))["summary"]
        cap = s.get("word_cap") or 350           # legacy loose-cap runs had no word_cap (=350)
        lab = s["label"]
        is_tr = "trained" in lab
        f = s.get("reward_frac")
        g = s.get("train_minus_held")
        if f is None or g is None:
            continue
        recs.append({"cap": cap, "f": f, "trained": is_tr, "gap": g})
    if not recs:
        _placeholder("fig_trap", "budget-trap pending"); print("[fig_trap] placeholder"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, fval, ttl in [(axes[0], 0.5, "$f{=}0.5$"), (axes[1], 0.25, "$f{=}0.25$")]:
        for tr, col, nm in [(False, "steelblue", "vanilla (control)"), (True, "crimson", "trained")]:
            pts = sorted([r for r in recs if r["trained"] == tr and abs(r["f"] - fval) < 1e-6],
                         key=lambda r: -r["cap"])
            if pts:
                ax.plot([r["cap"] for r in pts], [r["gap"] for r in pts],
                        marker="o", color=col, label=nm)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("page budget (words)"); ax.set_title(ttl, fontsize=9)
        ax.invert_xaxis()  # increasing budget pressure -> right
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("adaptivity gap $R_{train}-R_{held}$"); axes[0].legend(fontsize=7)
    fig.suptitle("Budget pressure gates the trap (loose=safe, tight=gamed)", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_trap.pdf")); plt.close()
    print("[fig_trap] ok")


def fig_hazard(hz="hazard_v2.json"):
    """Per-arm host-page-rewrite hazard coefficient with 95% CI: rewriting drives death for
    vanilla; mitigations attenuate it. Reads hazard_*.json."""
    a = _load(hz) or _load("hazard_prelim.json")
    if not a or not a.get("arms"):
        _placeholder("fig_hazard", "hazard pending"); print("[fig_hazard] placeholder"); return
    rows = []
    for arm, d in a["arms"].items():
        pr = d["coef"].get("page_rewrites")
        if pr:
            rows.append((NICE.get(arm, arm), pr["coef"], pr["ci"], pr["p"]))
    order = {"vanilla": 0, "conservative": 1, "anchored": 2, "trained (SFT)": 3, "trained (proto)": 4}
    rows.sort(key=lambda r: order.get(r[0], 9))
    plt.figure(figsize=(5.2, 3.3))
    ys = list(range(len(rows)))
    for y, (nm, c, ci, p) in zip(ys, rows):
        plt.errorbar(c, y, xerr=[[c - ci[0]], [ci[1] - c]], fmt="o",
                     color=("crimson" if p < 0.05 else "steelblue"), capsize=3)
        plt.annotate(f"p={p:.3f}", (c, y), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    plt.axvline(0, color="gray", lw=0.9)
    plt.yticks(ys, [r[0] for r in rows]); plt.xlabel("per-rewrite death hazard (logit coef, 95% CI)")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_hazard.pdf")); plt.close()
    print("[fig_hazard] ok")


def fig_goodhart():
    import numpy as np
    files = sorted(glob.glob(os.path.join(RES, "goodhart_*.json")))
    if not files:
        _placeholder("fig_goodhart", "Goodhart figure pending"); print("[fig_goodhart] placeholder"); return
    R = {json.load(open(f))["summary"]["label"]: json.load(open(f))["summary"] for f in files}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    # left: bars vanilla vs trained at f50 (headline mask)
    pick = [("vanilla_f50", "vanilla"), ("trained_f50", "trained ($f{=}0.5$)")]
    labels = ["$P_{train}$", "$P_{held}$", "$P_{fresh}$"]
    x = np.arange(3); w = 0.38
    shown = 0
    for i, (key, nm) in enumerate(pick):
        r = R.get(key)
        if not r:
            continue
        vals = [r["R_train"] or 0, r["R_held"] or 0, r["R_fresh"] or 0]
        ax[0].bar(x + (shown - 0.5) * w, vals, w, label=nm,
                  color=("steelblue" if "vanilla" in key else "crimson"))
        shown += 1
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("retention")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_title("probe-split retention", fontsize=9)
    # right: dose-response gap vs reward fraction (trained) + vanilla null
    tr = sorted([R[k] for k in R if k.startswith("trained")], key=lambda r: r.get("reward_frac") or 1)
    va = sorted([R[k] for k in R if k.startswith("vanilla")], key=lambda r: r.get("reward_frac") or 1)
    if tr:
        ax[1].plot([r["reward_frac"] for r in tr], [r["train_minus_held"] for r in tr],
                   marker="o", color="crimson", label="trained")
    if va:
        ax[1].plot([r["reward_frac"] for r in va], [r["train_minus_held"] for r in va],
                   marker="s", color="steelblue", ls="--", label="vanilla (null)")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_xlabel("rewarded fraction $f$"); ax[1].set_ylabel("$R_{train}-R_{held}$")
    ax[1].set_title("dose-response", fontsize=9); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_goodhart.pdf")); plt.close()
    print("[fig_goodhart] ok")


def fig_hoh(analysis="analysis_hoh.json"):
    a = _load(analysis)
    if not a:
        _placeholder("fig_hoh", "HoH figure pending"); print("[fig_hoh] placeholder"); return
    plt.figure(figsize=(5.2, 3.5))
    for tag, d in a["arms"].items():
        plt.plot(range(len(d["mean_R"])), d["mean_R"], marker="s", ms=3,
                 label=NICE.get(tag.replace("hohmc_", ""), tag))
    plt.xlabel("maintenance batch $t$")
    plt.ylabel("string-match retention (no judge)")
    plt.ylim(0, 1); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig_hoh.pdf")); plt.close()
    print("[fig_hoh] ok")


if __name__ == "__main__":
    fig_decay(); fig_cost(); fig_frontier(); fig_goodhart(); fig_hoh()
    fig_trap(); fig_hazard()
