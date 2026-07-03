"""Plot the forgetting curve from results/*.json -> figures/forgetting_curve.png
and print the headline numbers for the post."""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

STYLE = {  # arm_size -> (label, linestyle)
    "plain_large":   ("Large model, plain (rewrites)", "-"),
    "plain_small":   ("Small model, plain (rewrites)", "--"),
    "harness_large": ("Large model + rollback harness", "-"),
}


def load():
    out = {}
    for fn in glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(fn) == "summary.json":
            continue
        d = json.load(open(fn))
        key = f"{d['arm']}_{'small' if d.get('small') else 'large'}"
        out.setdefault(key, []).append(d)
    return out


def main():
    runs = load()
    if not runs:
        print("no results yet")
        return
    plt.figure(figsize=(7, 4.5))
    headline = {}
    for key, ds in sorted(runs.items()):
        # average over seeds
        T = min(len(d["log"]) for d in ds)
        xs = list(range(T))
        ys = [sum(d["log"][t]["retention"] for d in ds) / len(ds) for t in range(T)]
        label, ls = STYLE.get(key, (key, "-"))
        plt.plot(xs, [100 * y for y in ys], ls, marker="o", ms=3, label=label)
        headline[key] = {"r0": 100 * ys[0], "rT": 100 * ys[-1],
                         "rebuild": 100 * (sum(d.get("rebuild_retention", 0) for d in ds) / len(ds))}
    # rebuild reference line (from plain_large if present)
    ref = headline.get("plain_large", {}).get("rebuild")
    if ref:
        plt.axhline(ref, color="gray", ls=":", lw=1, label=f"Fresh rebuild ({ref:.0f}%)")
    plt.xlabel("Number of update rounds (rewrites)")
    plt.ylabel("Facts still retained (%)")
    plt.title("LLM-Wiki forgets facts as it rewrites itself\n(real Hermes-style wiki, SciFact-Open)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, "forgetting_curve.png")
    plt.savefig(out, dpi=150)
    print("saved", out)
    print("\n=== HEADLINE NUMBERS ===")
    for k, v in headline.items():
        print(f"{k}: start {v['r0']:.0f}% -> after updates {v['rT']:.0f}%  (fresh rebuild {v['rebuild']:.0f}%)")
    json.dump(headline, open(os.path.join(RES, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
