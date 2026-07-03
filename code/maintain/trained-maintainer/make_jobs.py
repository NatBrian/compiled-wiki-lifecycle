"""Trained-maintainer stage: generate the v2 experiment job manifest for run_pool.py.

Produces results/v2_jobs.json. Design:
  - v2 tournament (seeded sampling + track_incorp ON => honest variance AND a same-run
    preservation-incorporation frontier): vanilla/conservative/proto at 3 seeds, anchored &
    trained at 5 seeds (the contested comparison needing a CI).
  - currency: seeds 1,2 for vanilla + trained (seed0 already exists) => 3-seed error bars.
  - scale: 1.5B maintainer seeds 1,2 (seed0 exists) => a band on the catastrophic endpoint.
  - budget-trap evals: goodhart_eval at the tight cap for vanilla + trained x f in {50,25}.
Ports 8102/8103 are two 14B servers (GPU6/GPU7) each serving adapters b1, maintainer-lora,
tight-f50, tight-f25; 8106 is the 1.5B server (GPU7) for the scale check.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # bundled repo: no nested code/, scripts live here
PY = "python3"
PORTS = [8102, 8103]
jobs = []
i = 0


def port():
    global i
    p = PORTS[i % len(PORTS)]
    i += 1
    return p


def proto(name, result, cmd, env=None):
    jobs.append({"name": name, "result": result, "cmd": cmd, **({"env": env} if env else {})})


TRACK = "20"

# ---- v2 tournament ----------------------------------------------------------------
def tour(arm, maint, seeds, tag):
    for s in seeds:
        p = port()
        out = f"results/results_v2_{tag}_seed{s}.json"
        cmd = [PY, "p4_protocol.py", "--arm", arm, "--seed", str(s),
               "--samp_seed", str(s), "--track_incorp", TRACK,
               "--port", str(p), "--out", out]
        if maint:
            cmd += ["--maintainer_model", maint]
        proto(f"v2_{tag}_s{s}", out, cmd)


tour("vanilla", None, [0, 1, 2], "vanilla")
tour("conservative", None, [0, 1, 2], "conservative")
tour("anchored", None, [0, 1, 2, 3, 4], "anchored")
tour("vanilla", "b1", [0, 1, 2, 3, 4], "trained")
tour("vanilla", "maintainer-lora", [0, 1, 2], "proto")

# ---- currency (3 seeds: add 1,2) --------------------------------------------------
for s in [1, 2]:
    for label, maint in [("vanilla", None), ("b1", "b1")]:
        p = port()
        out = f"results/results_currency_{label}_seed{s}.json"
        cmd = [PY, "currency_safety.py", "--seed", str(s), "--port", str(p), "--out", out]
        if maint:
            cmd += ["--maintainer_model", maint]
        proto(f"curr_{label}_s{s}", out, cmd)

# ---- scale band (1.5B maintainer, add seeds 1,2; build/judge on 8102, maint on 8106) ----
for s in [1, 2]:
    out = f"results/results_scale_maint1p5b_seed{s}.json"
    cmd = [PY, "p4_protocol.py", "--arm", "vanilla", "--maintainer_model", "qwen2.5-1.5b",
           "--maint_port", "8106", "--seed", str(s), "--samp_seed", str(s),
           "--port", "8102", "--out", out]
    proto(f"scale1p5b_s{s}", out, cmd)

# ---- budget-trap evals (tight cap=120, deterministic replay) ----------------------
for f in ["50", "25"]:
    meta = f"results/sft_probe_tight_f{f}.meta.json"
    # vanilla policy replayed at the tight cap (matched control)
    pv = port()
    outv = f"results/goodhart_tight_vanilla_f{f}.json"
    proto(f"trap_van_f{f}", outv,
          [PY, "goodhart_eval.py", "--meta", meta, "--label", f"tight_vanilla_f{f}",
           "--word_cap", "120", "--port", str(pv), "--out", outv])
    # trained-under-tight-budget policy
    pt = port()
    outt = f"results/goodhart_tight_trained_f{f}.json"
    proto(f"trap_trn_f{f}", outt,
          [PY, "goodhart_eval.py", "--meta", meta, "--maintainer_model", f"tight-f{f}",
           "--label", f"tight_trained_f{f}", "--word_cap", "120", "--port", str(pt), "--out", outt])

out_path = os.path.join(P4, "results", "v2_jobs.json")
json.dump(jobs, open(out_path, "w"), indent=1)
print(f"{len(jobs)} jobs -> {out_path}")
for j in jobs:
    print(" ", j["name"], "->", j["result"])
