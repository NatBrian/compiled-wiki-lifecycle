"""Comprehensive refinement analysis -> results/analysis2.json.
  - A1 with seed CIs (mean +/- std over up to 5 seeds)
  - E2-rescue: large-probe judge-free HoH stream LCB (exact b_t) -> non-vacuous?
  - E4: gate around LoRA maintainer
  - E5: tau over-conservatism sweep
"""
import glob, json, os, sys, math, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
CODE = os.path.dirname(os.path.dirname(HERE))      # .../compiled-wiki-lifecycle/code
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
sys.path.insert(0, HERE)
from p5_stream import stream_lcb
RES = os.path.join(P5, "results")


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None, None
    return (round(sum(xs) / len(xs), 4), round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0)


def a1_with_ci():
    runs = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, "a1_*_seed*.json"))):
        d = json.load(open(f)); c = d["config"]; runs[c["arm"]][c["seed"]] = d
    out = {}
    for arm, seeds in runs.items():
        finals = [d["timeline"][-1] for d in seeds.values()]
        R = ms([f["valid"]["corrected"] for f in finals])
        inc = ms([f.get("incorp_rate") for f in finals])
        roll = ms([f.get("n_rolled", 0) for f in finals])
        out[arm] = {"n_seeds": len(seeds), "seeds": sorted(seeds),
                    "R12_mean": R[0], "R12_std": R[1],
                    "incorp_mean": inc[0], "incorp_std": inc[1],
                    "rolled_mean": roll[0]}
    return out


def e2_rescue():
    """Large-probe judge-free HoH: stream LCB with EXACT b_t (string-match, no margin)."""
    out = {}
    for d in load("e2big_hoh_*.json"):
        c = d["config"]; arm = c["arm"]; tl = d["timeline"]
        if not arm.startswith("lossgate_"):  # stream only meaningful for gated
            continue
        L0 = tl[0]["valid"]["raw"]
        pb = [{"d": tl[t]["destroyed_gate"], "n": tl[t]["n_conf_prev"]} for t in range(1, len(tl))]
        # exact (judge-free): tpr=1, fpr=0 -> margin 1 -> no inflation
        Lstream, _ = stream_lcb(pb, L0, 0.05, tpr=1.0, fpr=0.0, mode="confidence_sequence")
        realized = [tl[t]["valid"]["raw"] for t in range(len(tl))]
        nonvac = sum(1 for x in Lstream if x > 0)
        valid = all(realized[t] >= Lstream[t] - 1e-9 for t in range(len(Lstream)))
        out.setdefault(arm, []).append({
            "seed": c["seed"], "gate_n": c.get("n_probe", 0) // 2, "L0": round(L0, 4),
            "stream_lcb": [round(x, 4) for x in Lstream], "realized": [round(x, 4) for x in realized],
            "nonvacuous_batches": nonvac, "stream_valid": valid,
            "final_stream_lcb": round(Lstream[-1], 4), "final_realized": round(realized[-1], 4),
            "mean_bt": round(sum(p["d"] for p in pb) and 0 or 0, 4)})
    return out


def e4_lora():
    out = {}
    for d in load("e4_*_lora_seed*.json"):
        c = d["config"]; tl = d["timeline"]; f = tl[-1]
        gated = c["arm"].startswith("lossgate_")
        # coverage on valid pool
        hits = tot = 0
        for t in range(1, len(tl)):
            b = f and tl[t].get("b_t", 0.0)
            if gated:
                ok = tl[t]["valid"]["corrected"] >= tl[t-1]["valid"]["corrected"] - tl[t].get("b_t",0) - 1e-9
                hits += ok; tot += 1
        out[c["arm"] + "_lora"] = {"R12": f["valid"]["corrected"], "incorp": f.get("incorp_rate"),
                                   "rolled": f.get("n_rolled", 0),
                                   "coverage": (round(hits/tot,4) if tot else None)}
    return out


def e5_tau():
    out = {}
    for d in load("e5_lossgate_vanilla_tau*_seed0.json"):
        c = d["config"]; f = d["timeline"][-1]
        out[f"tau{c['tau']}"] = {"R12": f["valid"]["corrected"], "incorp": f.get("incorp_rate"),
                                 "rolled": f.get("n_rolled"), "b_t": f.get("b_t")}
    return out


def main():
    out = {"a1_ci": a1_with_ci(), "e2_rescue": e2_rescue(),
           "e4_lora": e4_lora(), "e5_tau": e5_tau()}
    json.dump(out, open(os.path.join(RES, "analysis2.json"), "w"), indent=1)
    print("=== A1 with CIs ===");
    for a, v in out["a1_ci"].items():
        print(f"  {a:24s} R12={v['R12_mean']}±{v['R12_std']} inc={v['incorp_mean']}±{v['incorp_std']} (n={v['n_seeds']})")
    print("=== E2-rescue (large-probe judge-free stream) ===")
    for a, runs in out["e2_rescue"].items():
        for r in runs:
            print(f"  {a} seed{r['seed']} gate_n={r['gate_n']}: stream_final={r['final_stream_lcb']} "
                  f"realized_final={r['final_realized']} nonvac_batches={r['nonvacuous_batches']}/13 valid={r['stream_valid']}")
    print("=== E4-LoRA ===", json.dumps(out["e4_lora"]))
    print("=== E5-tau ===", json.dumps(out["e5_tau"]))


if __name__ == "__main__":
    main()
