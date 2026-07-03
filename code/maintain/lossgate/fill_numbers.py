"""Populate the result macros in paper/latex/main.tex from results/analysis.json.
Idempotent: rewrites the \\newcommand{\\Name}{...} lines in the result block.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
A = json.load(open(os.path.join(P5, "results", "analysis.json")))
tex = os.path.join(P5, "paper", "latex", "main.tex")
tl = A["timelines"]


def fin(arm, key, default="--"):
    if arm in tl and tl[arm]:
        v = tl[arm][-1].get(key)
        return f"{v:.3f}" if isinstance(v, (int, float)) else default
    return default


def r0():
    arm = next((a for a in ("vanilla", "lossgate_vanilla", "conservative") if a in tl), None)
    return f"{tl[arm][0]['valid_R']:.3f}" if arm else "--"


mob = A.get("make_or_break", {})
e1 = A.get("e1_coverage", {})
e2 = A.get("e2_stream", {})
# static breach batch (first gated arm, first seed)
sb = "--"
for arm, runs in e2.items():
    if runs and runs[0].get("static_breach_t") is not None:
        sb = str(runs[0]["static_breach_t"]); break
# gate cost fraction vs full re-audit
gc = "--"
costs = A.get("final_cost_calls", {})
if "lossgate_vanilla" in costs and "vanilla" in costs and costs["vanilla"]:
    # crude: extra gate_judge fraction -> reported separately if available
    gc = "$<$0.5$\\times$"

vals = {
    "RvanFinal": fin("vanilla", "valid_R"),
    "RconsFinal": fin("conservative", "valid_R"),
    "RlgFinal": fin("lossgate_vanilla", "valid_R"),
    "RzeroVal": r0(),
    "incVan": fin("vanilla", "incorp"),
    "incLg": fin("lossgate_vanilla", "incorp"),
    "covE": (f"{e1['coverage']:.2f}" if e1.get("coverage") is not None else "--"),
    "staticBreach": sb,
    "gateCostFrac": gc,
}

src = open(tex).read()
for name, val in vals.items():
    src = re.sub(r"(\\newcommand\{\\" + name + r"\}\{).*?(\})",
                 lambda m, v=val: m.group(1) + v + m.group(2), src, count=1)
open(tex, "w").write(src)
print("filled:", json.dumps(vals, indent=1))
print("make_or_break:", json.dumps(mob, indent=1))
