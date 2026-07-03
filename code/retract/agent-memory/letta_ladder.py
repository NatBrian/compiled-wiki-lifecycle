"""Letta cross-paradigm replication of the agent-memory stage's retraction ladder.

Second memory PARADIGM (block-memory + archival/recall + autonomous
consolidation) to show the backflow + membership-veto result is not specific to
mem0's synthesized-note store.

Letta paradigm mapping (mirrors code/e0b_auto.py + code/b1_ladder.py semantics):
  - core memory BLOCK ("human" block)  = the SYNTHESIZED store the retraction targets
  - ARCHIVAL memory (vector store)      = the RETAINED reservoir (NOT purged by the
                                          block retraction) -- the analogue of the
                                          retained transcript
  - consolidation pass                  = an autonomous sleep-time step: re-read the
                                          retained archival passages and (re)write
                                          durable facts back into the block

Arms (headline diagonal):
  A1 : ingest fact into block + archival; retract from block only (archival kept);
       run K consolidation passes that re-derive from archival -> expect BACKFLOW.
  A3 : same, but a MEMBERSHIP WRITE-VETO gates the consolidation write -- any block
       content reconstructing the retracted value is removed -> expect RSR ~0.
  A4 : never-ingested oracle -- the d-supporting passages are replaced by
       entailment-neutral substitutes; d is never ingested -> RSR ~0.

We do NOT depend on Letta's agent loop / LLM tool-calling for the memory mechanics
(that would couple the result to Qwen3 tool-use reliability). We drive Letta's
PERSISTENCE primitives directly (blocks API + archival/passages API) and run the
SAME consolidation+veto+RSR logic as the mem0 ladder, so the only thing that
changes vs e0b_auto is the underlying memory substrate. The consolidation LLM
re-derivation step uses the shared vLLM (llm.ask), identical to the mem0 arms.

Output: results/letta_ladder.json
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import llm  # shared Qwen3-14B vLLM client (same as mem0 arms)

from letta_client import Letta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("P7_N_LETTA", "30"))
OFFSET = int(os.environ.get("P7_OFFSET", "0"))
N_CONSOLID = int(os.environ.get("P7_N_CONSOLID", "3"))
LETTA_URL = os.environ.get("LETTA_BASE_URL", "http://localhost:8285")
VLLM_URL = os.environ.get("P7_VLLM_URL", "http://localhost:8102/v1")
EMB_URL = os.environ.get("P7_EMB_URL", "http://localhost:8290/v1")
MODEL = os.environ.get("P7_MODEL", "Qwen/Qwen3-14B")

c = Letta(base_url=LETTA_URL)


# --- value-match predicate (identical to mem0_backend.contains_value) ---
def _norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def contains_value(text, value):
    return len(_norm(value)) >= 4 and _norm(value) in _norm(text)


# --- LLM + embedding config dicts pointing at the shared vLLM + CPU embed shim ---
# (letta_client 1.12.1 takes plain dicts; no exported config classes)
LLM_CONFIG = {
    "model": MODEL,
    "model_endpoint_type": "openai",
    "model_endpoint": VLLM_URL,
    "context_window": 8192,
}
EMB_CONFIG = {
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": EMB_URL,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": 384,
    "embedding_chunk_size": 300,
}


def make_agent(name):
    """Create a fresh Letta agent with an empty human block."""
    return c.agents.create(
        name=name,
        memory_blocks=[
            {"label": "human", "value": ""},
            {"label": "persona", "value": "I am a helpful memory assistant."},
        ],
        llm_config=LLM_CONFIG,
        embedding_config=EMB_CONFIG,
        include_base_tools=True,
    )


def get_block(agent_id, label="human"):
    return c.agents.blocks.retrieve(label, agent_id=agent_id)


def set_block(agent_id, value, label="human"):
    c.agents.blocks.update(label, agent_id=agent_id, value=value)


def append_block(agent_id, line, label="human"):
    cur = get_block(agent_id, label).value or ""
    set_block(agent_id, (cur + ("\n" if cur else "") + line), label)


def archival_insert(agent_id, text):
    c.agents.passages.create(agent_id, text=text)


def archival_all(agent_id):
    """Return list of (id, text) of all archival passages."""
    out, after = [], None
    while True:
        page = c.agents.passages.list(agent_id, limit=100, after=after)
        if not page:
            break
        out.extend([(p.id, p.text) for p in page])
        if len(page) < 100:
            break
        after = page[-1].id
    return out


def archival_search(agent_id, query, limit=5):
    """Semantic archival search -> list of passage texts (field is .content)."""
    try:
        r = c.agents.passages.search(agent_id, query=query)
        res = r.results if hasattr(r, "results") else []
        texts = [getattr(x, "content", None) or x.get("content")
                 if not hasattr(x, "content") else x.content for x in res]
        texts = [t for t in texts if t]
        if texts:
            return texts[:limit]
    except Exception:
        pass
    return [t for _, t in archival_all(agent_id)][:limit]


def block_has_value(agent_id, value, label="human"):
    return contains_value(get_block(agent_id, label).value or "", value)


def archival_delete_value(agent_id, value):
    """Membership-veto / retraction on the archival surfacing store: delete any
    passage that reconstructs the retracted value."""
    for pid, txt in archival_all(agent_id):
        if contains_value(txt, value):
            try:
                c.agents.passages.delete(pid, agent_id=agent_id)
            except Exception:
                pass


def consolidate(agent_id, transcript, value, veto):
    """One autonomous consolidation pass over the RETAINED transcript reservoir.

    Mirrors e0b_auto.consolidate: re-feed a window of the retained transcript
    (NOT purged by the retraction) and have the LLM re-derive durable facts, then
    write them back into BOTH surfacing stores (the core block + archival). If
    veto is set, any reconstruction of the retracted value is removed from BOTH
    stores after the write -- the membership write-veto on the consolidation hook.
    """
    window = "\n".join(transcript[-12:])
    cur_block = get_block(agent_id).value or ""
    prompt = (
        "You maintain a compact memory of durable facts. Below is the current "
        "memory, then a window of retained session log lines. Re-derive ALL "
        "durable facts (names, numbers, IDs, balances, conditions, preferences) "
        "supported by the log and produce the UPDATED memory as a short bullet "
        "list. Include every concrete fact found.\n\n"
        f"Current memory:\n{cur_block or '(empty)'}\n\n"
        f"Retained session log:\n{window}\n\nUpdated memory (bullets):")
    try:
        new_mem = llm.ask(prompt, temperature=0.0, max_tokens=400)
    except Exception:
        return
    # write the re-synthesized memory into BOTH surfacing stores
    set_block(agent_id, new_mem)
    for ln in new_mem.splitlines():
        ln = ln.strip("-* \t")
        if len(ln) > 3:
            try:
                archival_insert(agent_id, ln)
            except Exception:
                pass
    if veto:
        # membership write-veto on BOTH stores
        kept = [ln for ln in (get_block(agent_id).value or "").splitlines()
                if not contains_value(ln, value)]
        set_block(agent_id, "\n".join(kept))
        archival_delete_value(agent_id, value)


def grounded_answer(agent_id, question, value):
    """Answer grounded ONLY on the synthesized surfacing stores (block + archival
    retrieval) -- the analogue of mem0's mem.search over synthesized notes."""
    block = get_block(agent_id).value or ""
    notes = archival_search(agent_id, question, limit=5)
    ctx = "Memory block:\n" + (block or "(empty)") + "\n\nRetrieved notes:\n" + \
          ("\n".join(f"- {n}" for n in notes) if notes else "(none)")
    prompt = (f"{ctx}\n\nQuestion: {question}\nAnswer using ONLY the memory above. "
              f"If absent, reply UNKNOWN.")
    ans = llm.ask(prompt, temperature=0.0, max_tokens=60)
    return ans, contains_value(ans, value)


def delete_agent(agent_id):
    try:
        c.agents.delete(agent_id=agent_id)
    except Exception:
        pass


def run_arm(fact, arm):
    val = fact["value"]
    ent = fact["entity"]
    name = f"p7_{fact['id']}_{arm}_{int(time.time()*1000) % 100000}"
    a = make_agent(name)
    aid = a.id
    obs = {"arm": arm, "id": fact["id"], "value": val}
    try:
        # neutral substitutes for the oracle: never mention the value
        if arm == "A4":
            ingest = [
                f"{ent} had a routine appointment that went fine.",
                f"{ent} mentioned they were busy this week.",
                f"{ent} confirmed their contact details are on file.",
            ]
        else:
            ingest = [fact["statement"]] + fact.get("contexts", [])
        benign = f"{ent} enjoys weekend hiking."
        ingest = ingest + [benign]

        # RETAINED reservoir = a plain transcript (mirrors e0b_auto.transcript[]);
        # it is NOT purged by the retraction and is what consolidation re-reads.
        transcript = [f"user: {m}" for m in ingest]

        # SURFACING stores = core block + archival vector store. Seed both so the
        # retraction has something to target (the synthesized state the agent
        # answers from). Oracle seeds only the benign fact (secret never present).
        for m in ingest:
            seed = (arm != "A4") or (m == benign)
            if seed:
                append_block(aid, "- " + m)
                archival_insert(aid, m)

        # --- Retraction: remove the secret from BOTH surfacing stores
        #     (the retained transcript is kept) ---
        if arm in ("A1", "A3"):
            kept = [ln for ln in (get_block(aid).value or "").splitlines()
                    if not contains_value(ln, val)]
            set_block(aid, "\n".join(kept))
            archival_delete_value(aid, val)
        obs["in_block_after_delete"] = block_has_value(aid, val)
        obs["surfacing_after_delete"] = bool(block_has_value(aid, val) or
            any(contains_value(t, val) for _, t in archival_all(aid)))
        obs["transcript_retains"] = any(contains_value(t, val) for t in transcript)

        # --- Autonomous consolidation passes over the RETAINED transcript ---
        veto = (arm == "A3")
        for _ in range(N_CONSOLID):
            consolidate(aid, transcript, val, veto)

        in_block = block_has_value(aid, val)
        in_arch = any(contains_value(t, val) for _, t in archival_all(aid))
        ans, ans_leak = grounded_answer(aid, fact["question"], val)
        b_ans, b_ok = grounded_answer(aid, f"What does {ent} enjoy doing?", "hiking")
        obs["in_block_after_consolidation"] = in_block
        obs["in_archival_after_consolidation"] = in_arch
        obs["answer"] = ans[:140]
        obs["answer_leaked"] = ans_leak
        obs["backflowed"] = bool(in_block or in_arch or ans_leak)
        obs["benign_retained"] = bool(b_ok)
        obs["final_block"] = (get_block(aid).value or "")[:400]
    except Exception as e:
        obs["error"] = str(e)[:300]
    finally:
        delete_agent(aid)
    return obs


def _wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    cc = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((cc - h) / d, 4), round((cc + h) / d, 4)]


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    allf = json.load(open(f"{ROOT}/data/facts.json"))
    fs = allf[OFFSET:OFFSET + N]
    arms = ["A1", "A3", "A4"]
    print(f"[LETTA] {len(fs)} facts x {arms}, consolid={N_CONSOLID}, url={LETTA_URL}", flush=True)
    recs = []
    for i, f in enumerate(fs, 1):
        out = {"id": f["id"], "attr": f["attr"]}
        for arm in arms:
            try:
                out[arm] = run_arm(f, arm)
            except Exception as e:
                out[arm] = {"arm": arm, "error": str(e)[:300]}
        recs.append(out)
        # incremental save so partial progress survives
        json.dump({"per_fact": recs, "partial": True},
                  open(f"{ROOT}/results/letta_ladder.json", "w"), indent=2)
        print(f"[LETTA] {i}/{len(fs)} "
              f"A1bf={out['A1'].get('backflowed')} A3bf={out['A3'].get('backflowed')} "
              f"A4bf={out['A4'].get('backflowed')}", flush=True)

    summ = {"n_facts": len(recs), "n_consolid": N_CONSOLID,
            "paradigm": "letta_block+archival", "arms": {}}
    for arm in arms:
        xs = [r[arm] for r in recs if arm in r and "backflowed" in r[arm]]
        n = len(xs)
        bf = sum(x["backflowed"] for x in xs)
        br = sum(x.get("benign_retained", False) for x in xs)
        summ["arms"][arm] = {"n": n, "backflowed": bf,
                             "RSR": round(bf / n, 4) if n else None,
                             "RSR_wilson95": _wilson(bf, n),
                             "benign_retained": round(br / n, 4) if n else None}
    a1 = summ["arms"]["A1"]["RSR"]
    a3 = summ["arms"]["A3"]["RSR"]
    a4 = summ["arms"]["A4"]["RSR"]
    summ["backflow_exists"] = bool(a1 is not None and a1 >= 0.10)
    summ["gate_fixes"] = bool(a1 and a3 is not None and a3 <= a1 - 0.10)
    summ["a3_approx_oracle"] = bool(a3 is not None and a4 is not None and abs(a3 - a4) <= 0.10)
    summ["VERDICT"] = (
        "SUCCESS_cross_paradigm" if summ["backflow_exists"] and summ["gate_fixes"]
        else "NO_BACKFLOW" if not summ["backflow_exists"]
        else "BACKFLOW_BUT_GATE_WEAK")
    json.dump({"summary": summ, "per_fact": recs},
              open(f"{ROOT}/results/letta_ladder.json", "w"), indent=2)
    print("[LETTA] SUMMARY:", json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
