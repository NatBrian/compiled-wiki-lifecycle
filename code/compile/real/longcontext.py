"""Long-context controls to test what actually drives LLM-Wiki's low staleness:
is it the OVERWRITE/compile step, or just "small corpus fits in context"?

Faithful to Karpathy's LLM-Wiki, we DUMP the whole knowledge base into the reader
context (no top-k embedding retrieval) and ask the implicit-currency queries.

Conditions (anon Pydantic, current-fact-implicit, n=14):
  RAW_ALL   : dump every version of every concept (old + new). No curation.
  WIKI_ALL  : dump the compiled wiki pages (current symbol per concept, overwritten).
  RAW_CUR   : dump only the current-version docs (sanity floor).

If RAW_ALL leaks (high SER) but WIKI_ALL does not, the overwrite/compile step is
what kills staleness — not long context per se. If RAW_ALL also ~0, then dumping
everything suffices and Wiki's curation is not special.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("BENCH_ANON", "1")
from pathlib import Path
import arms as A
import bench_llm as L
import scorer

ROOT = Path(__file__).resolve().parent.parent
LIB = "pydantic"


def load():
    c = ROOT / "corpus" / LIB
    versions = json.loads((c/"manifest.json").read_text())["versions"]
    bundles = {v: [json.loads(l) for l in (c/f"v{v}.jsonl").read_text().splitlines() if l.strip()] for v in versions}
    gold = [json.loads(l) for l in (c/"gold.jsonl").read_text().splitlines() if l.strip()]
    imp = [q for q in gold if q["type"] == "current-fact-implicit"]
    return versions, bundles, imp


def read_and_score(context, queries):
    n = stale = cor = 0
    for q in queries:
        r = L.chat("You are a precise coding assistant. Use ONLY the context. "
                   "Answer with the exact API call or symbol only, nothing else.",
                   f"Context:\n{context}\n\nQuestion: {q['text']}", max_tokens=40)
        sc = scorer.score_factual(q["gold"], q.get("deprecated_answers", []), r.text)
        n += 1; stale += sc["stale"]; cor += sc["correct"]
    return {"n": n, "acc": round(cor/n, 3), "ser": round(stale/n, 3), "stale": stale}


def main():
    versions, bundles, imp = load()

    # RAW_ALL: every version of every concept (old + new), anonymized
    raw_all = "\n".join(A.doc_text(d) for v in versions for d in bundles[v])
    # RAW_CUR: only the latest version's docs (current-only floor) -- but concepts
    # introduced earlier and unchanged must be included; take latest doc per concept id
    latest = {}
    seq = 0
    for v in versions:
        for d in bundles[v]:
            latest[d["id"].split("__")[0]] = d
    raw_cur = "\n".join(A.doc_text(d) for d in latest.values())

    # WIKI_ALL: build the compiled wiki (overwrite per release), dump ALL pages
    wiki = A.LLMWiki()
    for v in versions:
        wiki.ingest(v, bundles[v])
    pages = []
    for cid, p in wiki.pages.items():
        hist = f" (was: {', '.join(p['history'])})" if p["history"] else ""
        pages.append(f"[{p['current']}] is the current way to {p['concept']}{hist}")
    wiki_all = "\n".join(pages)

    print(f"context sizes (chars): RAW_ALL={len(raw_all)} WIKI_ALL={len(wiki_all)} RAW_CUR={len(raw_cur)}")
    out = {}
    for name, ctx in [("RAW_ALL (dump every version)", raw_all),
                      ("WIKI_ALL (dump compiled wiki, Karpathy-faithful)", wiki_all),
                      ("RAW_CUR (dump current-only, floor)", raw_cur)]:
        res = read_and_score(ctx, imp)
        out[name] = res
        print(f"  {name:50} acc={res['acc']} SER={res['ser']} (stale={res['stale']}/{res['n']})")
    (ROOT/"results"/"results_longcontext_pyd.json").write_text(json.dumps(out, indent=2))
    print("wrote results_longcontext_pyd.json")


if __name__ == "__main__":
    main()
