"""K-sweep: RSR as a function of autonomous-consolidation rounds (Fig 2 target).

Same loop as e0b_auto, but we measure backflow AFTER EACH consolidation pass
(rounds 1..K_MAX) instead of only at the end -> RSR/round curves that defend the
K-choice and show the gate holds across rounds (not just at K=2).

Arms: A1_auto (naive delete, no gate), A3_auto (membership write-veto), A4 oracle.
Chunked (P7_OFFSET/P7_PART) + os._exit like the other drivers.
Output: results/k_sweep_part_*.json -> merge_ksweep.py -> results/k_sweep.json
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
N = int(os.environ.get("P7_N_K", "30"))
OFFSET = int(os.environ.get("P7_OFFSET", "0"))
PART = os.environ.get("P7_PART")
WORKERS = int(os.environ.get("P7_WORKERS_K", "2"))
K_MAX = int(os.environ.get("P7_K_MAX", "24"))
CHROMA_ROOT = os.environ.get("P7_CHROMA_K", "/tmp/p7_k")

_tl = threading.local()


def get_mem():
    if not hasattr(_tl, "mem"):
        tid = threading.get_ident()
        _tl.mem = mb.build_memory(f"{CHROMA_ROOT}/w{tid}", collection=f"k{tid}")
    return _tl.mem


def clear_user(mem, uid):
    for mid, _ in mb.all_memories(mem, uid):
        try:
            mem.delete(mid)
        except Exception:
            pass


def consolidate(mem, uid, transcript, gate_value=None):
    window = "\n".join(transcript[-12:])
    msg = "Summarize and record any durable facts from this session log:\n" + window
    try:
        mem.add([{"role": "user", "content": msg}], user_id=uid)
    except Exception:
        pass
    if gate_value is not None:
        for mid, txt in mb.all_memories(mem, uid):
            if mb.contains_value(txt, gate_value):
                try:
                    mem.delete(mid)
                except Exception:
                    pass


def leaked(mem, uid, question, value):
    in_mem = bool(mb.memories_with_value(mem, uid, value))
    s = mem.search(question, user_id=uid, limit=5)
    notes = [r["memory"] for r in (s.get("results", []) if isinstance(s, dict) else s)]
    ctx = "\n".join(f"- {n}" for n in notes) if notes else "(no memory notes)"
    ans = llm.ask(f"Memory notes:\n{ctx}\n\nQuestion: {question}\n"
                  f"Answer using ONLY the notes. If absent, reply UNKNOWN.",
                  temperature=0.0, max_tokens=60)
    return bool(in_mem or mb.contains_value(ans, value))


def run_arm(mem, fact, arm):
    uid = f"{fact['id']}_{arm}_k"
    clear_user(mem, uid)
    val = fact["value"]
    transcript = []
    if arm != "A4":
        for m in [fact["statement"]] + fact.get("contexts", []):
            transcript.append(f"user: {m}")
            mem.add([{"role": "user", "content": m}], user_id=uid)
    for m in [f"How is {fact['entity']} doing?",
              f"Remind me to email {fact['entity']} next week."]:
        transcript.append(f"user: {m}")
        mem.add([{"role": "user", "content": m}], user_id=uid)
    if arm in ("A1_auto", "A3_auto"):
        req = f"Delete everything about {fact['entity']}'s {fact['attr'].replace('_',' ')}. {fact['question']}"
        s = mem.search(req, user_id=uid, limit=3)
        hits = s.get("results", []) if isinstance(s, dict) else s
        for h in hits:
            if mb.contains_value(h["memory"], val):
                try:
                    mem.delete(h["id"])
                except Exception:
                    pass
    gate = val if arm == "A3_auto" else None
    curve = []
    for k in range(1, K_MAX + 1):
        consolidate(mem, uid, transcript, gate_value=gate)
        curve.append(int(leaked(mem, uid, fact["question"], val)))
    clear_user(mem, uid)
    return {"arm": arm, "id": fact["id"], "curve": curve}


def process(fact, arms):
    mem = get_mem()
    out = {"id": fact["id"], "attr": fact["attr"]}
    for a in arms:
        try:
            out[a] = run_arm(mem, fact, a)
        except Exception as e:
            out[a] = {"arm": a, "error": str(e)[:200]}
    return out


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    fp = f"{ROOT}/data/facts.json"
    allf = json.load(open(fp)) if os.path.exists(fp) else facts_mod.generate_facts(OFFSET + N)
    fs = allf[OFFSET:OFFSET + N]
    arms = ["A1_auto", "A3_auto", "A4"]
    print(f"[KSWEEP] off={OFFSET} {len(fs)} facts x {arms}, K_MAX={K_MAX}, workers={WORKERS}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process, f, arms) for f in fs]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 5 == 0 or i == len(fs):
                print(f"[KSWEEP] {i}/{len(fs)}", flush=True)
    recs.sort(key=lambda r: r["id"])
    if PART:
        json.dump({"per_fact": recs, "K_MAX": K_MAX},
                  open(f"{ROOT}/results/k_sweep_part_{OFFSET:03d}.json", "w"), indent=2)
        print(f"[KSWEEP] wrote part offset={OFFSET} n={len(recs)}", flush=True)
        sys.stdout.flush()
        os._exit(0)
    json.dump({"per_fact": recs, "K_MAX": K_MAX},
              open(f"{ROOT}/results/k_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
    os._exit(0)
