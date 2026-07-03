"""Min-K% / Min-K%++ membership-inference leakage (Block 1 Table 5, privacy).

Calibrated continuous leakage measure complementing the binary substring RSR.
For each fact we score the SECRET VALUE conditioned on the post-retraction
memory notes that each arm actually left behind (final_memories, already saved by
e0b_auto.py -> no mem0 rerun, only vLLM logprob calls):

  score(value | notes) via vLLM echo+logprobs:
    Min-K%   = mean of the bottom-K% target-token logprobs            (Shi'24)
    Min-K%++ = mean of the bottom-K% z-scored token logprobs, z from the
               per-position top-logprob distribution returned by vLLM   (Zhang'24, approx)

Higher score => value more predictable from the notes => more leaked/memorized.
MIA-AUC separates members (A1_auto, leaked store) from non-members (A4 oracle).
A well-gated store (A3_auto) should be ~indistinguishable from oracle (AUC~0.5).

Output: results/mia.json
"""
import json
import os
import sys

import numpy as np
import openai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("P7_VLLM_URL", "http://localhost:8102/v1")
MODEL = os.environ.get("P7_MODEL", "Qwen/Qwen3-14B")
KPCT = float(os.environ.get("P7_MINK", "0.2"))   # bottom 20%
N = int(os.environ.get("P7_N_MIA", "200"))
_c = openai.OpenAI(base_url=BASE, api_key="x")


def facts_by_id():
    fp = f"{ROOT}/data/facts.json"
    return {f["id"]: f for f in json.load(open(fp))} if os.path.exists(fp) else {}


def score(notes, entity, attr, value):
    """Return (mink, minkpp) for value conditioned on notes. None if unscorable."""
    ctx = "\n".join(f"- {n}" for n in notes) if notes else "(no memory notes)"
    pre = f"Memory notes:\n{ctx}\n\n{entity}'s {attr.replace('_',' ')} is "
    prompt = pre + value
    vstart = len(pre)
    try:
        r = _c.completions.create(model=MODEL, prompt=prompt, echo=True,
                                  max_tokens=0, logprobs=20, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return None, None, str(e)[:80]
    lp = r.choices[0].logprobs
    toks, tlp, off = lp.tokens, lp.token_logprobs, lp.text_offset
    top = lp.top_logprobs or [None] * len(toks)
    vals_lp, vals_z = [], []
    for i, (t, l, o) in enumerate(zip(toks, tlp, off)):
        if o is None or o < vstart or l is None:
            continue
        vals_lp.append(l)
        # approx mu/sigma from this position's returned top-logprob distribution
        d = top[i]
        if d and len(d) >= 2:
            arr = np.array(list(d.values()), dtype=float)
            mu, sd = arr.mean(), arr.std()
            vals_z.append((l - mu) / sd if sd > 1e-6 else 0.0)
        else:
            vals_z.append(0.0)
    if not vals_lp:
        return None, None, "no_value_tokens"
    k = max(1, int(len(vals_lp) * KPCT))
    mink = float(np.mean(sorted(vals_lp)[:k]))
    minkpp = float(np.mean(sorted(vals_z)[:k]))
    return mink, minkpp, None


def auc(pos, neg):
    """Rank-based ROC AUC: P(score(member) > score(nonmember))."""
    if not pos or not neg:
        return None
    pos, neg = np.array(pos), np.array(neg)
    wins = sum((pos[:, None] > neg[None, :]).sum() for _ in [0])
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def main():
    fbi = facts_by_id()
    d = json.load(open(f"{ROOT}/results/e0b_auto.json"))["per_fact"][:N]
    arms = ["A1_auto", "A3_auto", "A4"]
    rows = {a: {"mink": [], "minkpp": []} for a in arms}
    n_scored = 0
    for r in d:
        f = fbi.get(r["id"])
        if not f:
            continue
        ok = True
        tmp = {}
        for a in arms:
            arm = r.get(a, {})
            if "final_memories" not in arm:
                ok = False
                break
            mk, mpp, err = score(arm["final_memories"], f["entity"], f["attr"], f["value"])
            if mk is None:
                ok = False
                break
            tmp[a] = (mk, mpp)
        if not ok:
            continue
        for a in arms:
            rows[a]["mink"].append(tmp[a][0])
            rows[a]["minkpp"].append(tmp[a][1])
        n_scored += 1
        if n_scored % 20 == 0:
            print(f"[MIA] scored {n_scored}", flush=True)
    out = {"n_scored": n_scored, "k_pct": KPCT, "model": MODEL, "arms": {}}
    for a in arms:
        out["arms"][a] = {"mink_mean": round(float(np.mean(rows[a]["mink"])), 4) if rows[a]["mink"] else None,
                          "minkpp_mean": round(float(np.mean(rows[a]["minkpp"])), 4) if rows[a]["minkpp"] else None}
    out["MIA_AUC"] = {
        "A1_vs_A4_mink": auc(rows["A1_auto"]["mink"], rows["A4"]["mink"]),
        "A1_vs_A4_minkpp": auc(rows["A1_auto"]["minkpp"], rows["A4"]["minkpp"]),
        "A3_vs_A4_mink": auc(rows["A3_auto"]["mink"], rows["A4"]["mink"]),
        "A3_vs_A4_minkpp": auc(rows["A3_auto"]["minkpp"], rows["A4"]["minkpp"]),
    }
    out["interpretation"] = ("A1 vs oracle AUC>>0.5 = leaked store is MIA-detectable; "
                             "A3 vs oracle AUC~0.5 = gated store indistinguishable from never-ingested")
    json.dump(out, open(f"{ROOT}/results/mia.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    os._exit(0)
