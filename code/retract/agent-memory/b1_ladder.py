"""B1 -- the retraction ladder on the autonomous-consolidation loop.

Extends E0b-AUTO into the paper's core discrimination experiment. Same lifecycle
(retained transcript + synthesized mem0 + N autonomous consolidation passes), but
the consolidation WRITE HOOK varies by arm:

  A0  no retraction (control)                       -> RSR ~1.0
  A1  naive delete, no write gate                   -> RSR high (backflow)
  A2  delete + CORRECTNESS/faithfulness write-gate  -> RSR high  (KEY: a faithful
        re-derivation is allowed, so backflow persists -> correctness != deletion)
  A3  delete + MEMBERSHIP write-veto                -> RSR ~0    (the fix)
  A4  never-ingested oracle                          -> RSR ~0

Plus a benign-utility probe (over-veto / CFP): each persona also has a NON-secret
preference that is never retracted. A3 must keep RSR low on the secret WITHOUT
destroying benign memory (benign_retained should stay high, ~A1/A2 level).

Demonstrates: "stays correct != stays deleted" -- A2 (correctness) allows the
leak that A3 (membership) blocks -- discrimination = A2_RSR - A3_RSR.

Output: results/b1_ladder.json
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
N = int(os.environ.get("P7_N_B1", "40"))
OFFSET = int(os.environ.get("P7_OFFSET", "0"))
WORKERS = int(os.environ.get("P7_WORKERS_B1", "2"))
N_CONSOLID = int(os.environ.get("P7_N_CONSOLID", "2"))
CHROMA_ROOT = os.environ.get("P7_CHROMA_B1", "/tmp/p7_b1")
PART = os.environ.get("P7_PART")  # if set, write a per-chunk part file instead

_tl = threading.local()


def get_mem():
    if not hasattr(_tl, "mem"):
        tid = threading.get_ident()
        _tl.mem = mb.build_memory(f"{CHROMA_ROOT}/w{tid}", collection=f"b{tid}")
    return _tl.mem


def clear_user(mem, uid):
    for mid, _ in mb.all_memories(mem, uid):
        try:
            mem.delete(mid)
        except Exception:
            pass


def faithful_to_transcript(memory_text, transcript):
    """Correctness/faithfulness gate predicate: is memory_text supported by the log?"""
    log = "\n".join(transcript[-14:])
    prompt = (f"Session log:\n{log}\n\nStatement: {memory_text}\n\n"
              f"Is the statement supported by the log? Answer YES or NO only.")
    return llm.ask(prompt, temperature=0.0, max_tokens=6).strip().upper().startswith("YES")


def consolidate(mem, uid, transcript, arm, retracted_value):
    """One autonomous consolidation pass with the arm's write hook."""
    window = "\n".join(transcript[-12:])
    try:
        mem.add([{"role": "user", "content":
                  "Summarize and record any durable facts from this session log:\n" + window}],
                user_id=uid)
    except Exception:
        return
    if arm == "A2":
        # correctness gate: keep faithful memories, drop only unfaithful ones.
        # A faithful re-derivation of the secret is KEPT -> backflow persists.
        for mid, txt in mb.all_memories(mem, uid):
            if not faithful_to_transcript(txt, transcript):
                try:
                    mem.delete(mid)
                except Exception:
                    pass
    elif arm == "A3":
        # membership write-veto: drop anything reconstructing the retracted value.
        for mid, txt in mb.all_memories(mem, uid):
            if mb.contains_value(txt, retracted_value):
                try:
                    mem.delete(mid)
                except Exception:
                    pass


def grounded_answer(mem, uid, question, needle):
    s = mem.search(question, user_id=uid, limit=5)
    notes = [r["memory"] for r in (s.get("results", []) if isinstance(s, dict) else s)]
    ctx = "\n".join(f"- {n}" for n in notes) if notes else "(no memory notes)"
    ans = llm.ask(f"Memory notes:\n{ctx}\n\nQuestion: {question}\nAnswer using ONLY the "
                  f"notes. If absent, reply UNKNOWN.", temperature=0.0, max_tokens=60)
    return ans, mb.contains_value(ans, needle)


def run_arm(mem, fact, arm):
    uid = f"{fact['id']}_{arm}"
    clear_user(mem, uid)
    val = fact["value"]
    benign = f"{fact['entity']} enjoys weekend hiking."
    benign_q = f"What does {fact['entity']} enjoy doing?"
    benign_needle = "hiking"
    transcript = []

    if arm != "A4":
        for m in [fact["statement"]] + fact.get("contexts", []) + [benign]:
            transcript.append(f"user: {m}")
            mem.add([{"role": "user", "content": m}], user_id=uid)
    else:
        # oracle: only the benign fact is ever ingested (secret never told)
        transcript.append(f"user: {benign}")
        mem.add([{"role": "user", "content": benign}], user_id=uid)

    obs = {"arm": arm, "id": fact["id"], "value": val}

    # retraction (synthesized memory only)
    if arm in ("A1", "A2", "A3"):
        req = f"Delete everything about {fact['entity']}'s {fact['attr'].replace('_',' ')}. {fact['question']}"
        s = mem.search(req, user_id=uid, limit=3)
        for h in (s.get("results", []) if isinstance(s, dict) else s):
            if mb.contains_value(h["memory"], val):
                try:
                    mem.delete(h["id"])
                except Exception:
                    pass

    for _ in range(N_CONSOLID):
        consolidate(mem, uid, transcript, arm, val)

    in_mem = bool(mb.memories_with_value(mem, uid, val))
    ans, ans_leak = grounded_answer(mem, uid, fact["question"], val)
    b_ans, b_ok = grounded_answer(mem, uid, benign_q, benign_needle)
    obs["backflowed"] = bool(in_mem or ans_leak)
    obs["benign_retained"] = bool(b_ok)
    obs["answer"] = ans[:120]
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
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    fp = f"{ROOT}/data/facts.json"
    allf = json.load(open(fp)) if os.path.exists(fp) else facts_mod.generate_facts(OFFSET + N)
    fs = allf[OFFSET:OFFSET + N]
    arms = ["A0", "A1", "A2", "A3", "A4"]
    print(f"[B1] {len(fs)} facts x {arms}, consolid={N_CONSOLID}, workers={WORKERS}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process, f, arms) for f in fs]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 5 == 0 or i == len(fs):
                print(f"[B1] {i}/{len(fs)}", flush=True)
    recs.sort(key=lambda r: r["id"])
    summ = {"n_facts": len(recs), "n_consolid": N_CONSOLID, "arms": {}}
    for a in arms:
        xs = [r[a] for r in recs if a in r and "backflowed" in r[a]]
        n = len(xs); bf = sum(x["backflowed"] for x in xs); br = sum(x["benign_retained"] for x in xs)
        summ["arms"][a] = {"n": n,
                           "RSR": round(bf / n, 4) if n else None, "RSR_wilson95": _wilson(bf, n),
                           "benign_retained": round(br / n, 4) if n else None}
    a2 = summ["arms"]["A2"]["RSR"]; a3 = summ["arms"]["A3"]["RSR"]
    summ["discrimination_A2_minus_A3"] = round((a2 or 0) - (a3 or 0), 4)
    summ["membership_not_correctness"] = bool(a2 and a3 is not None and (a2 - a3) >= 0.10)
    TAG = os.environ.get("P7_TAG", "")
    if PART:
        json.dump({"per_fact": recs}, open(f"{ROOT}/results/b1{TAG}_part_{OFFSET:03d}.json", "w"), indent=2)
        print(f"[B1] wrote part offset={OFFSET} n={len(recs)}", flush=True)
    else:
        json.dump({"summary": summ, "per_fact": recs}, open(f"{ROOT}/results/b1_ladder.json", "w"), indent=2)
        print("[B1] SUMMARY:", json.dumps(summ, indent=2), flush=True)
    sys.stdout.flush()
    os._exit(0)  # force exit: mem0/chroma leak non-daemon threads that hang shutdown


if __name__ == "__main__":
    main()
