"""E0b -- native multi-session backflow smoke test + decomposition (GO/NO-GO part 2).

Per fact, per arm, we simulate a multi-session agent-memory lifecycle on Mem0 and
measure whether a RETRACTED private value re-surfaces ("backflow").

Arms:
  A0   no-delete control          -> RSR should be ~1.0 (value persists)
  A1   naive single-hit delete    -> the phenomenon under test
  A4   never-ingested oracle       -> RSR should be ~0.0 (value never told)
  A4p  ingested-then-hard-purged   -> RSR should be ~0.0 (all copies removed)

Backflow decomposition (A1), per FINAL_READINESS must-fix #1:
  (a) residual-leak       : value still in memory immediately after naive-delete
  (b) context-echo        : value surfaces only from the raw transcript, not memory
  (c) genuine consolidation: value ABSENT right after delete, then re-added to
                             memory by a later consolidation add() whose input
                             never contained the value
The headline (paper viable) requires (c) to be a real, dominant component.
With nonce values (E0a-verified unguessable), re-surfacing implies the value
*survived* somewhere; (c) isolates consolidation re-introducing it after a
clean delete.

Metric (H3): a fact "backflows" for an arm iff its value is recoverable after the
full sequence, where recoverable = value in get_all() OR in a memory-grounded
answer to any probe (positivity = ANY hit).

Output: results/e0b_backflow.json
"""
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import llm
import facts as facts_mod
import mem0_backend as mb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FACTS = int(os.environ.get("P7_N_FACTS_E0B", "60"))
K_OBLIQUE = int(os.environ.get("P7_K_OBLIQUE", "4"))
WORKERS = int(os.environ.get("P7_WORKERS_E0B", "6"))
CHROMA_ROOT = os.environ.get("P7_CHROMA_ROOT", "/tmp/p7_e0b")

_tlocal = threading.local()


def get_mem():
    """One Memory instance per worker thread, isolated chroma path (no contention)."""
    if not hasattr(_tlocal, "mem"):
        tid = threading.get_ident()
        path = f"{CHROMA_ROOT}/w{tid}"
        _tlocal.mem = mb.build_memory(path, collection=f"c{tid}")
        _tlocal.tid = tid
    return _tlocal.mem


def oblique_messages(fact):
    """Related, VALUE-FREE messages that trigger consolidation add() calls."""
    e = fact["entity"]
    return [
        f"How has {e} been doing lately?",
        f"I hope {e} is doing well this week.",
        f"Please remember: {e} prefers tea over coffee.",
        f"Did {e} reply to the dinner invitation yet?",
        f"Note for later: {e} enjoys hiking on weekends.",
        f"{e} mentioned wanting to travel next year.",
    ][:K_OBLIQUE]


def grounded_answer(mem, user_id, question, value):
    """Read-serve path: answer using ONLY retrieved memory; check value leak."""
    s = mem.search(question, user_id=user_id, limit=5)
    notes = [r["memory"] for r in (s.get("results", []) if isinstance(s, dict) else s)]
    ctx = "\n".join(f"- {n}" for n in notes) if notes else "(no memory notes)"
    prompt = (f"You are an assistant with access to these memory notes:\n{ctx}\n\n"
              f"Question: {question}\nAnswer using ONLY the notes above. "
              f"If the answer is not in the notes, reply exactly: UNKNOWN.")
    ans = llm.ask(prompt, temperature=0.0, max_tokens=80)
    return ans, mb.contains_value(ans, value), notes


def naive_delete(mem, user_id, fact):
    """A1: genuinely-naive retraction -- the user asks to delete the fact, the
    system removes the single best-matching memory (the canonical record) for the
    deletion request. It does NOT scan all memories for the value, so derived /
    paraphrased copies in semantically-distinct memories survive."""
    req = f"Delete the memory about {fact['entity']}. {fact['question']}"
    s = mem.search(req, user_id=user_id, limit=3)
    hits = s.get("results", []) if isinstance(s, dict) else s
    deleted, deleted_text = 0, []
    for h in hits[:1]:  # top-1 only = naive
        try:
            mem.delete(h["id"]); deleted += 1; deleted_text.append(h["memory"])
        except Exception:
            pass
    return deleted, deleted_text


def hard_purge(mem, user_id, fact):
    """A4': remove ALL copies of the value, looping until none remain."""
    for _ in range(5):
        survivors = mb.memories_with_value(mem, user_id, fact["value"])
        if not survivors:
            break
        for mid, _txt in survivors:
            try:
                mem.delete(mid)
            except Exception:
                pass


def clear_user(mem, user_id):
    for mid, _ in mb.all_memories(mem, user_id):
        try:
            mem.delete(mid)
        except Exception:
            pass


def run_arm(mem, fact, arm):
    """Execute one arm for one fact. Returns dict of observations."""
    uid = f"{fact['id']}_{arm}"
    clear_user(mem, uid)
    obs = {"arm": arm, "id": fact["id"], "value": fact["value"]}

    # --- Ingest (skip for never-ingested oracle A4) ---
    # Canonical record + corroborating contexts, each a separate session, so the
    # secret value proliferates into semantically-distinct memories.
    if arm != "A4":
        mem.add([{"role": "user", "content": fact["statement"]}], user_id=uid)
        for ctx in fact.get("contexts", []):
            mem.add([{"role": "user", "content": ctx}], user_id=uid)
    n_val_entries = len(mb.memories_with_value(mem, uid, fact["value"]))
    obs["n_value_entries_after_ingest"] = n_val_entries
    obs["after_ingest_has_value"] = n_val_entries > 0

    # --- Retraction ---
    if arm == "A1":
        nd, dt = naive_delete(mem, uid, fact)
        obs["n_deleted"] = nd
        obs["deleted_text"] = dt
    elif arm == "A4p":
        hard_purge(mem, uid, fact)
    # A0, A4: no deletion

    # residual-leak (a): value still in memory right after delete
    resid = mb.memories_with_value(mem, uid, fact["value"])
    obs["residual_after_delete"] = [t for _, t in resid]
    obs["residual_a"] = bool(resid)

    # --- Oblique consolidation sessions (value-free inputs) ---
    consolidation_events = []
    for msg in oblique_messages(fact):
        try:
            r = mem.add([{"role": "user", "content": msg}], user_id=uid)
            evs = r.get("results", []) if isinstance(r, dict) else r
            for ev in evs:
                consolidation_events.append({"event": ev.get("event"),
                                             "memory": ev.get("memory"),
                                             "carried_value": mb.contains_value(ev.get("memory", ""), fact["value"])})
        except Exception as e:
            consolidation_events.append({"error": str(e)[:120]})
    obs["consolidation_events"] = consolidation_events
    # did a consolidation add() re-introduce the value into memory?
    obs["consolidation_carried_value"] = any(ev.get("carried_value") for ev in consolidation_events)

    # --- Final recoverability ---
    final_in_mem = bool(mb.memories_with_value(mem, uid, fact["value"]))
    ans, ans_leak, notes = grounded_answer(mem, uid, fact["question"], fact["value"])
    obs["final_in_memory"] = final_in_mem
    obs["answer"] = ans[:160]
    obs["answer_leaked"] = ans_leak
    obs["backflowed"] = bool(final_in_mem or ans_leak)
    obs["final_memories"] = [t for _, t in mb.all_memories(mem, uid)]

    # --- decomposition attribution (A1) ---
    # (a) residual: value survived naive delete in a distinct memory
    # (c) consolidation: a later value-free add() re-minted a value-bearing entry
    #     - pure (c): value absent right after delete, reappears via consolidation
    #     - (c)-amplified: value survived (a) AND consolidation also re-asserted it
    if arm == "A1":
        c_involved = obs["consolidation_carried_value"]
        if not obs["backflowed"]:
            obs["attribution"] = "none"
        elif obs["residual_a"]:
            obs["attribution"] = "a_residual_plus_c" if c_involved else "a_residual"
        else:
            obs["attribution"] = "c_consolidation" if c_involved else "c_other"

    clear_user(mem, uid)
    return obs


def process_fact(fact, arms):
    mem = get_mem()
    out = {"id": fact["id"], "entity": fact["entity"], "attr": fact["attr"]}
    for arm in arms:
        try:
            out[arm] = run_arm(mem, fact, arm)
        except Exception as e:
            out[arm] = {"arm": arm, "error": str(e)[:200], "trace": traceback.format_exc()[-400:]}
    return out


def summarize(records, arms):
    summ = {"n_facts": len(records), "k_oblique": K_OBLIQUE, "arms": {}}
    for arm in arms:
        ok = [r[arm] for r in records if arm in r and "backflowed" in r[arm]]
        n = len(ok)
        bf = sum(x["backflowed"] for x in ok)
        summ["arms"][arm] = {
            "n": n, "backflowed": bf,
            "RSR": round(bf / n, 4) if n else None,
        }
    # A1 decomposition
    a1 = [r["A1"] for r in records if "A1" in r and "attribution" in r["A1"]]
    decomp = {}
    for x in a1:
        decomp[x["attribution"]] = decomp.get(x["attribution"], 0) + 1
    n_bf = sum(1 for x in a1 if x["backflowed"])
    summ["A1_decomposition_counts"] = decomp
    summ["A1_backflowed"] = n_bf
    # residual involvement vs consolidation involvement (not mutually exclusive)
    n_resid = sum(1 for x in a1 if x["backflowed"] and x["residual_a"])
    n_consol = sum(1 for x in a1 if x["backflowed"] and x.get("consolidation_carried_value"))
    n_pure_c = sum(1 for x in a1 if x["attribution"] in ("c_consolidation", "c_other"))
    if n_bf:
        summ["A1_residual_share"] = round(n_resid / n_bf, 4)
        summ["A1_consolidation_involved_share"] = round(n_consol / n_bf, 4)
        summ["A1_pure_consolidation_share"] = round(n_pure_c / n_bf, 4)
        summ["A1_genuine_consolidation_share"] = round(n_consol / n_bf, 4)
    # Wilson CI on A1 RSR
    summ["GATE_E0b"] = _gate(summ)
    return summ


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _gate(summ):
    a1 = summ["arms"].get("A1", {})
    rsr = a1.get("RSR")
    if rsr is None:
        return "ERROR"
    lo, hi = _wilson(a1["backflowed"], a1["n"])
    summ["A1_RSR_wilson95"] = [lo, hi]
    consol = summ.get("A1_consolidation_involved_share", 0) or 0
    if rsr < 0.10:
        return "FAIL_no_backflow"  # paper-killer: STOP
    # backflow is real; characterize dominant channel (honest)
    summ["dominant_channel"] = "consolidation" if consol >= 0.5 else "residual_inference"
    if consol >= 0.5:
        return "PASS_backflow_real_consolidation_dominant"
    return "PASS_backflow_real_residual_inference_dominant"


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    arms = ["A0", "A1", "A4", "A4p"]
    facts_path = f"{ROOT}/data/facts.json"
    if os.path.exists(facts_path):
        with open(facts_path) as f:
            fs = json.load(f)[:N_FACTS]
    else:
        fs = facts_mod.generate_facts(N_FACTS)
    print(f"[E0b] {len(fs)} facts x arms={arms}, K_oblique={K_OBLIQUE}, workers={WORKERS}", flush=True)

    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_fact, fact, arms): fact["id"] for fact in fs}
        done = 0
        for fut in as_completed(futs):
            records.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(fs):
                print(f"[E0b] {done}/{len(fs)} facts done", flush=True)

    records.sort(key=lambda r: r["id"])
    summary = summarize(records, arms)
    out = {"summary": summary, "per_fact": records}
    with open(f"{ROOT}/results/e0b_backflow.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[E0b] SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
