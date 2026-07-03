"""Audit-set sizing curve: how the certified bound tightens with audit size n.
Resamples the E1 wiki audit verdicts at varying n (no new GPU). Makes the
"price of certification" claim concrete. Outputs figs/sizing.pdf + json.
"""
import json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from stats import retention_lcb

e1 = json.load(open(os.path.join(HERE, "results_e1.json")))
v_wiki = e1["verdicts"]["wiki"]
vv = e1["verdicts"]
cal = {"k_tpr": sum(vv["tpr_single"]), "n_tpr": len(vv["tpr_single"]),  # single-gold n=40
       "k_fpr": sum(vv["fpr_wiki"]), "n_fpr": len(vv["fpr_wiki"])}
rhat = e1["report"]["wiki"]["corrected_R"]

rng = random.Random(0)
ns = [10, 15, 20, 30, 40, 50, 60, 76]
rows = []
for n in ns:
    lcbs = []
    for _ in range(2000):
        sub = [v_wiki[i] for i in (rng.randrange(len(v_wiki)) for _ in range(n))]
        lcb, _ = retention_lcb(sum(sub), n, cal["k_tpr"], cal["n_tpr"],
                               cal["k_fpr"], cal["n_fpr"], alpha=0.05)
        lcbs.append(lcb)
    lcbs.sort()
    rows.append({"n": n, "mean_LCB": sum(lcbs) / len(lcbs),
                 "p10": lcbs[len(lcbs) // 10], "p90": lcbs[9 * len(lcbs) // 10]})
    print(f"n={n}: mean LCB {rows[-1]['mean_LCB']:.3f} "
          f"[{rows[-1]['p10']:.3f},{rows[-1]['p90']:.3f}]  price={rhat-rows[-1]['mean_LCB']:.3f}")

json.dump({"rhat": rhat, "rows": rows}, open(os.path.join(HERE, "results_sizing.json"), "w"), indent=2)

fig, ax = plt.subplots(figsize=(5, 3.2))
xs = [r["n"] for r in rows]
mean = [r["mean_LCB"] for r in rows]
ax.fill_between(xs, [r["p10"] for r in rows], [r["p90"] for r in rows],
                color="#2166ac", alpha=0.15, label="10--90\\% over resamples")
ax.plot(xs, mean, "o-", color="#2166ac", label="certificate LCB")
ax.axhline(rhat, color="#404040", ls="--", lw=0.8, label=f"corrected $\\hat{{R}}$={rhat:.2f}")
ax.set_xlabel("audit-set size $n$ (facts sampled)")
ax.set_ylabel("certified retention LCB")
ax.set_title("Price of certification shrinks with audit size")
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "paper", "latex", "figs", "sizing.pdf"))
print("wrote sizing.pdf")
