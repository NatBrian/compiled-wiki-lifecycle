"""Long-context control on HoH (model does NOT know these Wikipedia facts).

For each item, DUMP all evidence passages into the reader context (no retrieval,
no curation) and ask the question. Compare:
  RAW_ALL : dump every evidence passage (outdated + current).
  RAW_CUR : dump only the current evidence (what overwrite/resolve effectively give).

If RAW_ALL leaks (SER >> RAW_CUR) on these unknown facts, then long-context dump
does NOT solve staleness when the model can't self-disambiguate -> curation /
recency genuinely matters (defends the Wiki/resolve result). If RAW_ALL ~ RAW_CUR,
dumping everything suffices and curation is not special.

Run: real/venvs/<any>/bin/python real/hoh_longcontext.py [N]  (uses ls_test deps)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def run(condition):
    items = real_core.load_hoh(N)
    rows = []
    for i, rec in items:
        docs, query = real_core.hoh_stream(i, rec)
        if condition == "RAW_ALL":
            ctx = "\n\n".join(d["_text"] for d in docs)                  # all versions
        else:  # RAW_CUR
            ctx = docs[-1]["_text"]                                      # current only
        out = real_core.hoh_read(ctx, query["text"])
        sc = real_core.hoh_score(out["answer"], query["gold"], query["deprecated_answers"])
        rows.append({"gold": query["gold"], "resp": out["answer"][:60], **sc,
                     "n_dep": len(query["deprecated_answers"])})
    return real_core.hoh_summarize(rows), rows


def main():
    import json
    res = {}
    for cond in ["RAW_ALL", "RAW_CUR"]:
        s, rows = run(cond)
        res[cond] = s
        print(f"  {cond:10} acc={s['accuracy']} SER={s['ser']} (stale={s['n_stale']}/{s['n']})", flush=True)
    (real_core.RESULTS / "results_hoh_longcontext.json").write_text(json.dumps(res, indent=2))
    print("wrote results_hoh_longcontext.json")


if __name__ == "__main__":
    main()
