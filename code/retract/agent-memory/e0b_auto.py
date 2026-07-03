"""E0b-AUTO -- autonomous-consolidation backflow (the paper's HEADLINE vector).

Real agent-memory systems keep a RETAINED reservoir (raw conversation / recall
log) and periodically RE-CONSOLIDATE it into synthesized memory (sleep-time
agent, summarizer, OpenClaw active-memory loop). If a retraction deletes the fact
from the *synthesized* store but the *retained transcript* still contains it, the
next autonomous consolidation re-derives and re-writes it -> BACKFLOW.

This models that loop on Mem0 explicitly:
  - transcript[]      = retained raw messages (NOT purged by the retraction)
  - mem0              = synthesized memory (what the retraction targets)
  - consolidation     = re-run add() over a transcript window (autonomous)

Arms:
  A1_auto : naive delete from synthesized memory, then autonomous consolidation
            over the retained transcript  -> expected RSR high  (backflow)
  A3_auto : same, but a MEMBERSHIP WRITE-VETO gates consolidation -- any newly
            written memory that reconstructs the retracted value is removed
            -> expected RSR ~0  (the weight-free fix)
  A4      : never-ingested oracle                                  -> RSR ~0

Output: results/e0b_auto.json
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import llm
import facts as facts_mod
import mem0_backend as mb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("P7_N_AUTO", "30"))
OFFSET = int(os.environ.get("P7_OFFSET", "0"))
PART = os.environ.get("P7_PART")  # if set, write a per-chunk part file instead
WORKERS = int(os.environ.get("P7_WORKERS_AUTO", "3"))
N_CONSOLID = int(os.environ.get("P7_N_CONSOLID", "2"))   # autonomous passes
CHROMA_ROOT = os.environ.get("P7_CHROMA_AUTO", "/tmp/p7_auto")

_tl = threading.local()


def get_mem():
    if not hasattr(_tl, "mem"):
        tid = threading.get_ident()
        _tl.mem = mb.build_memory(f"{CHROMA_ROOT}/w{tid}", collection=f"a{tid}")
    return _tl.mem


def clear_user(mem, uid):
    for mid, _ in mb.all_memories(mem, uid):
        try:
            mem.delete(mid)
        except Exception:
            pass


def consolidate(mem, uid, transcript, gate_value=None):
    """One autonomous consolidation pass: re-feed a window of the RETAINED
    transcript through add() (mem0 re-extracts + ADD/UPDATE). If gate_value is
    set, apply a membership write-veto: delete any resulting memory that contains
    the retracted value (the weight-free fix at the write/consolidate hook)."""
    window = "\n".join(transcript[-12:])
    msg = ("Summarize and record any durable facts from this session log:\n" + window)
    try:
        mem.add([{"role": "user", "content": msg}], user_id=uid)
    except Exception:
        pass
    if gate_value is not None:
        # write-veto: remove any memory whose text reconstructs the retracted value
        for mid, txt in mb.all_memories(mem, uid):
            if mb.contains_value(txt, gate_value):
                try:
                    mem.delete(mid)
                except Exception:
                    pass


def grounded_answer(mem, uid, question, value):
    s = mem.search(question, user_id=uid, limit=5)
    notes = [r["memory"] for r in (s.get("results", []) if isinstance(s, dict) else s)]
    ctx = "\n".join(f"- {n}" for n in notes) if notes else "(no memory notes)"
    prompt = (f"Memory notes:\n{ctx}\n\nQuestion: {question}\n"
              f"Answer using ONLY the notes. If absent, reply UNKNOWN.")
    ans = llm.ask(prompt, temperature=0.0, max_tokens=60)
    return ans, mb.contains_value(ans, value)


def run_arm(mem, fact, arm):
    uid = f"{fact['id']}_{arm}"
    clear_user(mem, uid)
    val = fact["value"]
    transcript = []

    # --- Session 1+: ingest fact + corroborating contexts into BOTH stores ---
    if arm != "A4":
        for m in [fact["statement"]] + fact.get("contexts", []):
            transcript.append(f"user: {m}")
            mem.add([{"role": "user", "content": m}], user_id=uid)
    # a few benign oblique turns (also retained)
    for m in [f"How is {fact['entity']} doing?",
              f"Remind me to email {fact['entity']} next week."]:
        transcript.append(f"user: {m}")
        mem.add([{"role": "user", "content": m}], user_id=uid)

    obs = {"arm": arm, "id": fact["id"], "value": val}

    # --- Retraction: delete from SYNTHESIZED memory only (transcript retained) ---
    if arm in ("A1_auto", "A3_auto"):
        req = f"Delete everything about {fact['entity']}'s {fact['attr'].replace('_',' ')}. {fact['question']}"
        s = mem.search(req, user_id=uid, limit=3)
        hits = s.get("results", []) if isinstance(s, dict) else s
        # naive but value-aware delete of matching synthesized memories
        for h in hits:
            if mb.contains_value(h["memory"], val):
                try:
                    mem.delete(h["id"])
                except Exception:
                    pass
    obs["in_mem_after_delete"] = bool(mb.memories_with_value(mem, uid, val))

    # --- Autonomous consolidation passes over the RETAINED transcript ---
    gate = val if arm == "A3_auto" else None
    for _ in range(N_CONSOLID):
        consolidate(mem, uid, transcript, gate_value=gate)

    # --- Recoverability ---
    in_mem = bool(mb.memories_with_value(mem, uid, val))
    ans, ans_leak = grounded_answer(mem, uid, fact["question"], val)
    obs["in_mem_after_consolidation"] = in_mem
    obs["answer"] = ans[:140]
    obs["answer_leaked"] = ans_leak
    obs["backflowed"] = bool(in_mem or ans_leak)
    obs["final_memories"] = [t for _, t in mb.all_memories(mem, uid)]
    clear_user(mem, uid)
    return obs


def process(fact, arms):
    mem = get_mem()
    out = {"id": fact["id"], "attr": fact["attr"]}
    for a in arms:
        try:
            out[a] = run_arm(mem, fact, a)
        except Exception as e:
            out[a] = {"arm": a, "error": str(e)[:200]}
    return out


def _wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    fp = f"{ROOT}/data/facts.json"
    allf = json.load(open(fp)) if os.path.exists(fp) else facts_mod.generate_facts(OFFSET + N)
    fs = allf[OFFSET:OFFSET + N]
    arms = ["A1_auto", "A3_auto", "A4"]
    print(f"[AUTO] off={OFFSET} {len(fs)} facts x {arms}, consolid passes={N_CONSOLID}, workers={WORKERS}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process, f, arms) for f in fs]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 5 == 0 or i == len(fs):
                print(f"[AUTO] {i}/{len(fs)}", flush=True)
    recs.sort(key=lambda r: r["id"])

    # Chunk mode: write a part file and hard-exit (mem0/chroma leak non-daemon threads)
    if PART:
        json.dump({"per_fact": recs}, open(f"{ROOT}/results/e0b_auto_part_{OFFSET:03d}.json", "w"), indent=2)
        print(f"[AUTO] wrote part offset={OFFSET} n={len(recs)}", flush=True)
        sys.stdout.flush()
        os._exit(0)
    summ = {"n_facts": len(recs), "n_consolid": N_CONSOLID, "arms": {}}
    for a in arms:
        xs = [r[a] for r in recs if a in r and "backflowed" in r[a]]
        n = len(xs); bf = sum(x["backflowed"] for x in xs)
        summ["arms"][a] = {"n": n, "backflowed": bf,
                           "RSR": round(bf / n, 4) if n else None,
                           "wilson95": _wilson(bf, n)}
    a1 = summ["arms"]["A1_auto"]; a3 = summ["arms"]["A3_auto"]
    summ["backflow_exists"] = bool(a1["RSR"] and a1["RSR"] >= 0.10)
    summ["gate_fixes"] = bool(a1["RSR"] and a3["RSR"] is not None and a3["RSR"] <= a1["RSR"] - 0.10)
    summ["GATE_AUTO"] = ("PASS_backflow_and_gate" if summ["backflow_exists"] and summ["gate_fixes"]
                         else "PASS_backflow_only" if summ["backflow_exists"]
                         else "FAIL_no_backflow")
    json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/e0b_auto.json", "w"), indent=2)
    print("[AUTO] SUMMARY:", json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # force exit: mem0/chroma leak non-daemon threads that hang shutdown
