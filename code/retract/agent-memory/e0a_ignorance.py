"""E0a -- backbone-ignorance audit (the agent-memory stage's GO/NO-GO gate, part 1).

Goal: verify the synthetic private facts are NOT already known to Qwen3-14B.
If the cold model can produce a fact's secret value, that fact is a confound
(we could not attribute later 'backflow' to memory). Such facts are discarded.

Pass criterion (per fact): cold leakage == 0 over all paraphrases x temps.
We also record abstention rate (model should say 'I don't know', not hallucinate
a confident specific value) as a dataset-quality signal.

Output:
  data/facts.json              -- full generated dataset
  results/e0a_ignorance.json   -- per-fact audit + summary
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import llm
import facts as facts_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FACTS = int(os.environ.get("P7_N_FACTS", "250"))
TEMPS = [0.0, 0.7]
WORKERS = int(os.environ.get("P7_WORKERS", "16"))

_ABSTAIN_PAT = re.compile(
    r"\b(i (don'?t|do not) (have|know)|no (information|record|data)|"
    r"cannot|can'?t (provide|find|help)|unable to|not aware|don'?t have access|"
    r"fictional|made[- ]?up|no (such )?(person|individual)|i'?m not able|"
    r"there is no|i have no)\b", re.IGNORECASE)


def _norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def value_leaked(answer, value):
    """True if the secret value appears in the answer (normalized substring)."""
    return _norm(value) in _norm(answer) and len(_norm(value)) >= 4


def audit_fact(fact):
    leaks = 0
    abstains = 0
    n = 0
    examples = []
    for probe in fact["paraphrases"]:
        for t in TEMPS:
            ans = llm.ask(probe, temperature=t, max_tokens=128)
            n += 1
            if value_leaked(ans, fact["value"]):
                leaks += 1
                examples.append({"probe": probe, "temp": t, "answer": ans[:200]})
            if _ABSTAIN_PAT.search(ans):
                abstains += 1
    return {
        "id": fact["id"], "entity": fact["entity"], "attr": fact["attr"],
        "n_probes": n, "leaks": leaks, "abstains": abstains,
        "cold_accuracy": leaks / n, "abstain_rate": abstains / n,
        "passes": leaks == 0, "leak_examples": examples,
    }


def main():
    os.makedirs(f"{ROOT}/data", exist_ok=True)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    fs = facts_mod.generate_facts(N_FACTS)
    with open(f"{ROOT}/data/facts.json", "w") as f:
        json.dump(fs, f, indent=2)
    print(f"[E0a] generated {len(fs)} facts -> data/facts.json", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(audit_fact, fact): fact["id"] for fact in fs}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"[E0a] audited {done}/{len(fs)}", flush=True)

    results.sort(key=lambda r: r["id"])
    n_pass = sum(r["passes"] for r in results)
    summary = {
        "n_facts": len(results),
        "n_pass_ignorance": n_pass,
        "pass_rate": n_pass / len(results),
        "mean_cold_accuracy": sum(r["cold_accuracy"] for r in results) / len(results),
        "mean_abstain_rate": sum(r["abstain_rate"] for r in results) / len(results),
        "n_temps": len(TEMPS), "temps": TEMPS,
        "failing_facts": [r["id"] for r in results if not r["passes"]],
        "GATE_E0a": "PASS" if n_pass >= 250 else (
            "PARTIAL" if n_pass >= 0.9 * len(results) else "FAIL"),
    }
    out = {"summary": summary, "per_fact": results}
    with open(f"{ROOT}/results/e0a_ignorance.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[E0a] SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
