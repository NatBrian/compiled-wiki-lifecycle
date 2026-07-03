"""Analyze A1 + E1/E2 from results/a1_*.json.

Outputs results/analysis.json with:
  - per-arm timelines averaged across seeds (valid retention, b_t, incorp, cost, rollback)
  - A1 make-or-break verdict: lossgate vs conservative on (loss, currency, cost) + freeze check
  - E1 certificate coverage: P(R_valid(t) >= R_valid(t-1) - b_t) over gated (arm,seed,t)
  - E2 stream composition: anytime-valid stream LCB vs P2 static certificate breach
"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P5 = HERE                                          # bundled repo: no nested code/, scripts live here
CODE = os.path.dirname(os.path.dirname(HERE))      # .../llm-wiki/code
sys.path.insert(0, os.path.join(CODE, "shared"))   # oai_client/maintain/certify/stats/currency/data
sys.path.insert(0, HERE)
from p5_stream import stream_lcb
RES = os.path.join(P5, "results")


def load_runs():
    runs = defaultdict(dict)  # arm -> seed -> data
    for f in sorted(glob.glob(os.path.join(RES, "a1_*.json"))):
        d = json.load(open(f))
        c = d["config"]
        runs[c["arm"]][c["seed"]] = d
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def avg_timeline(seed_runs):
    """Average per-t fields across seeds for one arm."""
    T = min(len(d["timeline"]) for d in seed_runs.values())
    out = []
    for t in range(T):
        rows = [d["timeline"][t] for d in seed_runs.values()]
        import statistics as _st
        _vR = [r["valid"]["corrected"] for r in rows]
        _inc = [r["incorp_rate"] for r in rows if r.get("incorp_rate") is not None]
        out.append({
            "t": t,
            "valid_R_std": round(_st.pstdev(_vR), 4) if len(_vR) > 1 else 0.0,
            "incorp_std": round(_st.pstdev(_inc), 4) if len(_inc) > 1 else 0.0,
            "valid_R": mean([r["valid"]["corrected"] for r in rows]),
            "valid_lcb": mean([r["valid"]["lcb"] for r in rows]),
            "valid_raw": mean([r["valid"]["raw"] for r in rows]),
            "gate_R": mean([r["gate"]["corrected"] for r in rows]),
            "b_t": mean([r.get("b_t", 0.0) for r in rows]),
            "incorp": mean([r.get("incorp_rate") for r in rows]),
            "n_committed": mean([r.get("n_committed", 0) for r in rows]),
            "n_rolled": mean([r.get("n_rolled", 0) for r in rows]),
            "store_words": mean([r.get("store_words") for r in rows]),
        })
    return out


def total_cost(d):
    g = d["timeline"][-1]["gen_calls"]
    return {k: g.get(k, 0) for k in ("build", "maint", "gate_judge", "audit_judge")}, sum(g.values())


def e1_coverage(runs, alpha=0.05):
    """Per-batch certificate coverage on the held-out VALID pool, gated arms only."""
    hits = total = 0
    detail = []
    for arm, seeds in runs.items():
        if not arm.startswith("lossgate_"):
            continue
        for seed, d in seeds.items():
            tl = d["timeline"]
            for t in range(1, len(tl)):
                b = tl[t].get("b_t", 0.0)
                rv_prev = tl[t - 1]["valid"]["corrected"]
                rv_now = tl[t]["valid"]["corrected"]
                ok = rv_now >= rv_prev - b - 1e-9
                hits += ok; total += 1
                detail.append({"arm": arm, "seed": seed, "t": t, "b_t": round(b, 4),
                               "drop": round(rv_prev - rv_now, 4), "ok": ok})
    return {"coverage": round(hits / total, 4) if total else None, "n": total,
            "nominal": 1 - alpha, "detail": detail}


def e2_stream(runs, delta=0.05):
    """Stream composition for gated arms; compare to P2 static cert (L0 forever)."""
    out = {}
    for arm, seeds in runs.items():
        if not arm.startswith("lossgate_"):
            continue
        per_arm = []
        for seed, d in seeds.items():
            tl = d["timeline"]
            cal = d["cal"]
            tpr = cal["k_tpr"] / cal["n_tpr"]; fpr = cal["k_fpr"] / cal["n_fpr"]
            L0 = tl[0]["valid"]["lcb"]
            pb = [{"d": tl[t].get("destroyed_gate", 0), "n": tl[t].get("n_conf_prev", 0)}
                  for t in range(1, len(tl))]
            Lstream, _ = stream_lcb(pb, L0, delta, tpr, fpr, mode="confidence_sequence")
            realized = [tl[t]["valid"]["corrected"] for t in range(len(tl))]
            # P2 static cert: L0 held forever -> breaches when realized < L0
            static_breach = next((t for t in range(len(realized)) if realized[t] < L0 - 1e-9), None)
            stream_breach = next((t for t in range(len(Lstream))
                                  if realized[t] < Lstream[t] - 1e-9), None)
            per_arm.append({"seed": seed, "L0": round(L0, 4),
                            "stream_lcb": [round(x, 4) for x in Lstream],
                            "realized": [round(x, 4) for x in realized],
                            "static_breach_t": static_breach,
                            "stream_breach_t": stream_breach})
        out[arm] = per_arm
    return out


def make_or_break(arms):
    """lossgate_vanilla vs conservative vs vanilla at final t."""
    def final(arm):
        tl = arms.get(arm)
        return tl[-1] if tl else None
    v, c, lg = final("vanilla"), final("conservative"), final("lossgate_vanilla")
    lgc = final("lossgate_conservative")
    verdict = {}
    if v and c and lg:
        # currency: incorporation rate (freeze detector). retention: valid_R at t=12.
        verdict = {
            "vanilla":      {"valid_R": v["valid_R"], "incorp": v["incorp"], "rolled": v["n_rolled"]},
            "conservative": {"valid_R": c["valid_R"], "incorp": c["incorp"], "rolled": c["n_rolled"]},
            "lossgate_vanilla": {"valid_R": lg["valid_R"], "incorp": lg["incorp"], "rolled": lg["n_rolled"]},
            "lossgate_conservative": ({"valid_R": lgc["valid_R"], "incorp": lgc["incorp"],
                                       "rolled": lgc["n_rolled"]} if lgc else None),
        }
        # PASS if lossgate_vanilla retains >= conservative AND keeps currency (incorp not collapsed
        # vs vanilla) -> i.e. it does NOT win by freezing.
        beats_conservative = lg["valid_R"] >= (c["valid_R"] or 0) - 1e-9
        currency_ok = (lg["incorp"] is not None and v["incorp"] is not None
                       and lg["incorp"] >= 0.7 * v["incorp"])
        verdict["beats_conservative_retention"] = bool(beats_conservative)
        verdict["currency_preserved_not_frozen"] = bool(currency_ok)
        verdict["PASS"] = bool(beats_conservative and currency_ok)
    return verdict


def main():
    runs = load_runs()
    if not runs:
        print("no results yet"); return
    arms = {arm: avg_timeline(seeds) for arm, seeds in runs.items()}
    costs = {arm: {seed: total_cost(d) for seed, d in seeds.items()} for arm, seeds in runs.items()}
    out = {
        "arms_present": {a: sorted(s.keys()) for a, s in runs.items()},
        "timelines": arms,
        "final_cost_calls": {a: mean([total_cost(d)[1] for d in s.values()])
                             for a, s in runs.items()},
        "make_or_break": make_or_break(arms),
        "e1_coverage": e1_coverage(runs),
        "e2_stream": e2_stream(runs),
    }
    json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1)
    print("=== A1 final (t=12) ===")
    for a, tl in arms.items():
        f = tl[-1]
        print(f"  {a:24s} validR={f['valid_R']} incorp={f['incorp']} "
              f"rolled/b={f['n_rolled']:.1f} b_t={f['b_t']}")
    print("=== make-or-break ===")
    print(json.dumps(out["make_or_break"], indent=1))
    print("=== E1 coverage ===", out["e1_coverage"]["coverage"], "n=", out["e1_coverage"]["n"],
          "nominal", out["e1_coverage"]["nominal"])
    print("=== cost (mean calls) ===", {a: round(v) for a, v in out["final_cost_calls"].items() if v})


if __name__ == "__main__":
    main()
