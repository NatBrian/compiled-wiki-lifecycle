"""Assemble the real-systems + resolver + closed-book results into one table
with Wilson 95% CIs on the implicit-staleness rate (n=14, so CIs are wide and we
report them honestly). Writes results/results_realsystems_pydantic.json + prints
the markdown table.
"""
import json, math, os, sys, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(max(0, c-h), 3), round(min(1, c+h), 3))


def imp_stats(rows):
    imp = [r for r in rows if r["type"] == "current-fact-implicit"]
    n = len(imp); stale = sum(r["stale"] for r in imp); cor = sum(r["correct"] for r in imp)
    return n, stale, cor


def closed_book():
    """Recompute + persist the no-retrieval parametric baseline."""
    from openai import OpenAI
    c = OpenAI(base_url=os.environ.get("AGENT_URL","http://127.0.0.1:8101/v1"), api_key="dummy")
    versions, bundles, by_version = real_core.load_corpus("pydantic")
    rows = []
    for v in versions:
        for q in by_version.get(v, []):
            if q["type"] == "synthesis": continue
            r = c.chat.completions.create(model=os.environ.get("AGENT_MODEL","qwen2.5-14b"),
                temperature=0.0, max_tokens=40,
                messages=[{"role":"system","content":"You are a precise coding assistant. Answer with the exact API call or symbol only, nothing else."},
                          {"role":"user","content":q["text"]}])
            a = r.choices[0].message.content or ""
            sc = real_core.score_factual(q["gold"], q.get("deprecated_answers",[]), a)
            rows.append({"qid":q["qid"],"type":q["type"],"version":v,"gold":q["gold"],
                         "answer":a,"correct":sc["correct"],"stale":sc["stale"],"qtokens":0})
    real_core.write_results("closedbook_pyd", "Closed-book (no retrieval)", rows,
                            extra={"prediction":"parametric prior is stale"})
    return rows


def load_rows(tag):
    p = R / f"results_{tag}.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return d["rows"]


def main():
    # closed-book (persist if missing)
    cb = load_rows("closedbook_pyd")
    if cb is None:
        print("[assemble] computing closed-book baseline...")
        cb = closed_book()

    # our arms from the a6val run + prior anon run
    a6 = json.loads((R/"results_a6val_pyd.json").read_text())["arms"]
    anon = json.loads((R/"results_anon_pyd_14b.json").read_text())["arms"]

    table = []  # (system, kind, mechanism, rows-or-stats)
    def add(name, kind, mech, rows):
        n, stale, cor = imp_stats(rows)
        lo, hi = wilson(stale, n)
        table.append({"system":name, "kind":kind, "mechanism":mech,
                      "n":n, "stale":stale, "ser_implicit":round(stale/n,3) if n else None,
                      "ser_ci":[lo,hi], "impl_acc":round(cor/n,3) if n else None})

    add("Closed-book (no retrieval)", "none", "parametric prior", cb)
    add("Mem0 (real)", "memory", "consolidate, recency-blind", load_rows("real_mem0_pydantic"))
    add("Vector RAG (ours)", "RAG", "accumulate", a6["A1_vector_rag"]["rows"])
    add("LightRAG (real)", "graph-RAG", "accumulate", load_rows("real_lightrag_pydantic"))
    add("Agent-memory (ours)", "memory", "append-only accumulate", anon["A3_agent_memory"]["rows"])
    add("RAPTOR (real)", "hierarchical-RAG", "accumulate (tree)", load_rows("real_raptor_pydantic"))
    add("A6 label-free resolve (ours)", "resolver", "recency-aware resolve (no oracle)", a6["A6_resolve_labelfree"]["rows"])
    add("LLM-Wiki (ours)", "wiki", "overwrite", anon["A4_llm_wiki"]["rows"])
    add("GraphRAG-lite (ours)", "graph", "supersession chain", anon["A2_graph_rag"]["rows"])
    add("A5 oracle resolve (ours)", "resolver", "resolve w/ gold labels", a6["A5_superseded_rag"]["rows"])

    # sort by SER desc
    table.sort(key=lambda r: (-(r["ser_implicit"] or 0)))
    out = {"dataset":"pydantic-anon", "metric":"SER_implicit (current-fact-implicit, n=14)", "rows":table}
    (R/"results_realsystems_pydantic.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'system':<32}{'kind':<18}{'SER_i':>7}{'95% CI':>16}{'acc':>7}  mechanism")
    print("-"*98)
    for r in table:
        ci = f"[{r['ser_ci'][0]:.2f},{r['ser_ci'][1]:.2f}]" if r['ser_ci'][0] is not None else "n/a"
        print(f"{r['system']:<32}{r['kind']:<18}{r['ser_implicit']:>7.3f}{ci:>16}{r['impl_acc']:>7.2f}  {r['mechanism']}")
    print(f"\nwrote {R/'results_realsystems_pydantic.json'}")


if __name__ == "__main__":
    main()
