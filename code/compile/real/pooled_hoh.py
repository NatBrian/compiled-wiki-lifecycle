"""Pooled-HoH: ONE shared corpus from ALL n HoH items (not per-item isolation).

Why: per-item HoH gives each query only its own 2-3 evidence docs (~66 tok), so
full-dump == wiki trivially (everything fits, nothing to disambiguate). Pooling
every item's evidence into a single corpus (628 docs for n=300) forces each system
to RETRIEVE / CURATE the right current fact out of a large mixed pool that also
contains stale versions of this AND other questions. This is the currency-at-scale
crossover: does curation (wiki/resolve) still beat dump/RAG when the corpus is big?

Faithfulness:
  closed_book   : parametric prior, no context.
  full_dump_cic : LOFT Corpus-in-Context; ALL docs ID-tagged in context, no curation.
                  Uses the long-context reader (256K Qwen3-Coder on :8102).
  vector_rag    : dense bge retrieval (top-k) over the full pool; stale coexists.
  wiki_karpathy : per-article wiki page holding ONLY the current symbol (overwritten
                  in ingest order); FTS5 keyword navigation (NO embeddings); read top
                  pages. Stale version never exists on the page.
  resolve_free  : label-free resolver on the RAG pool: cluster by text similarity,
                  keep latest-INGESTED per cluster (no gold id/version). Deployable.

Run:  real/venvs/lightrag/bin/python real/pooled_hoh.py <arm> [N]
arms: closed_book full_dump_cic vector_rag wiki_karpathy resolve_free  (or 'all')
"""
import os, sys, sqlite3, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import real_core

N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ARM = sys.argv[1] if len(sys.argv) > 1 else "all"

TAGSFX = os.environ.get("TAG_SUFFIX", "")   # e.g. "_1p5b" for the reader-size sweep
DUMP_MODEL = os.environ.get("DUMP_MODEL", "qwen3-coder-30b")
DUMP_URL = os.environ.get("DUMP_URL", "http://127.0.0.1:8102/v1")
RAG_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")
RAG_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
K = int(os.environ.get("POOL_K", "4"))


# ---------------------------------------------------------------- build the pool
STAMP = os.environ.get("BENCH_STAMP", "0") == "1"   # version-STAMPED regime (vs anonymous)
STATIC = os.environ.get("BENCH_STATIC", "0") == "1" # static corpus: current-only, no superseded

def build_pool():
    """Return (docs, queries). docs: global list with a global ingest_seq giving a
    single timeline (per concept, current = last). queries: one per item.

    BENCH_STAMP=1 serves each doc with an explicit version/recency marker (the
    latest-ingested doc per concept = CURRENT, earlier = OUTDATED). This is the
    *stamped* regime of §4.1: with the label visible, even accumulating arms can
    read off the current fact, so the staleness contrast collapses. Default is the
    version-ANONYMOUS regime (no markers) used for all headline results."""
    items = real_core.load_hoh(N)
    docs, queries = [], []
    gseq = 0
    for cid, (i, rec) in enumerate(items):
        stream, query = real_core.hoh_stream(i, rec)
        block = []
        for d in stream:
            d = dict(d)
            d["concept_id"] = cid            # = Wikipedia article boundary (page key)
            d["ingest_seq"] = gseq; gseq += 1
            block.append(d)
            docs.append(d)
        if STAMP and block:
            last = block[-1]["ingest_seq"]
            for k, d in enumerate(block):
                mark = ("[CURRENT — most recent update]" if d["ingest_seq"] == last
                        else f"[OUTDATED — superseded, version {k+1} of {len(block)}]")
                d["_text"] = f"{mark} {d['_text']}"
        query["concept_id"] = cid
        queries.append(query)
    if STATIC:
        # static-corpus regime (§4.6 on-stack): keep ONLY the current (latest-ingested)
        # doc per concept -- nothing is ever superseded, so no arm can be stale. Isolates
        # whether compilation (wiki) beats retrieval (RAG) on FIDELITY when currency is
        # not at stake. Re-index ingest_seq densely so retrieval is unaffected.
        latest = {}
        for d in docs:
            if d["concept_id"] not in latest or d["ingest_seq"] > latest[d["concept_id"]]["ingest_seq"]:
                latest[d["concept_id"]] = d
        docs = sorted(latest.values(), key=lambda x: x["concept_id"])
        for s, d in enumerate(docs):
            d["ingest_seq"] = s
    return docs, queries


# ---------------------------------------------------------------- readers
from openai import OpenAI
_clients = {}
def _client(url):
    if url not in _clients:
        _clients[url] = OpenAI(base_url=url, api_key="dummy")
    return _clients[url]

def read_ctx(context, question, model, url, system):
    r = _client(url).chat.completions.create(
        model=model, temperature=0.0, max_tokens=40,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return (r.choices[0].message.content or "").strip()

SYS_CTX = ("You answer a factual question using ONLY the context. The context may "
           "contain conflicting statements from different times; give the CURRENT "
           "answer. Reply with just the answer, as short as possible.")
SYS_CIC = ("You are given a CORPUS of ID-tagged documents. Use ONLY the corpus. It "
           "may contain outdated documents from different times; identify and give "
           "the CURRENT answer. Reply with just the answer, as short as possible.")
SYS_CB = ("Answer the factual question with your best current knowledge. "
          "Reply with just the answer, as short as possible.")


# ---------------------------------------------------------------- arms
def arm_closed_book(docs, queries, tag):
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        ans = read_ctx("(none)", q["text"], RAG_MODEL, RAG_URL, SYS_CB)
        _emit(tag, rows, q, ans)
    return rows

def arm_full_dump_cic(docs, queries, tag):
    corpus = "\n".join(f"[{j}] {d['_text']}" for j, d in enumerate(docs))
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        ans = read_ctx(corpus, q["text"], DUMP_MODEL, DUMP_URL, SYS_CIC)
        _emit(tag, rows, q, ans)
    return rows

def arm_vector_rag(docs, queries, tag):
    from retriever import make_retriever
    ret = make_retriever()
    ret.add(docs)
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        hits = ret.search(q["text"], K)
        ctx = "\n".join(h["_text"] for h in hits) or "(no documents)"
        ans = read_ctx(ctx, q["text"], RAG_MODEL, RAG_URL, SYS_CTX)
        _emit(tag, rows, q, ans, extra={"ctx_syms": [h["symbol"] for h in hits]})
    return rows

def arm_resolve_free(docs, queries, tag):
    from retriever import make_retriever
    import resolver
    ret = make_retriever()
    ret.add(docs)
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        pool = ret.search(q["text"], K * 3)                       # wider pool
        kept = resolver.resolve_chunks(pool)[:K]                  # cluster+latest seq
        ctx = "\n".join(h["_text"] for h in kept) or "(no documents)"
        ans = read_ctx(ctx, q["text"], RAG_MODEL, RAG_URL, SYS_CTX)
        _emit(tag, rows, q, ans, extra={"ctx_syms": [h["symbol"] for h in kept]})
    return rows

def _entity_graph(docs):
    """Structured store: entity nodes keyed by concept_id (the article boundary a
    graph-RAG would extract as an entity), each node carrying its version chain
    ordered by ingest_seq (the supersedes-edge ordering)."""
    by_cid = {}
    for d in docs:
        by_cid.setdefault(d["concept_id"], []).append(d)
    for cid in by_cid:
        by_cid[cid].sort(key=lambda x: x["ingest_seq"])
    return by_cid

def _graph_nodes(ret, query, by_cid):
    """Retrieve with the SAME dense retriever as the flat arms, then collapse hits
    to their entity nodes (rank-stable). Identical retrieval for both graph arms;
    only the per-node serving (accumulate vs resolve) differs."""
    hits = ret.search(query, K * 3)
    seen = []
    for h in hits:
        if h["concept_id"] in by_cid and h["concept_id"] not in seen:
            seen.append(h["concept_id"])
    return seen[:K]

def arm_graph_accumulate(docs, queries, tag):
    """Structured (entity-graph) + ACCUMULATE: serve every version on each retrieved
    entity node (mirrors LightRAG concatenating an entity's descriptions). Structure
    present, no resolution -> expected to leak."""
    from retriever import make_retriever
    ret = make_retriever(); ret.add(docs)
    by_cid = _entity_graph(docs)
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        nodes = _graph_nodes(ret, q["text"], by_cid) or [q["concept_id"]]
        parts = [d["_text"] for c in nodes for d in by_cid[c]]   # ALL versions
        ctx = "\n".join(parts) or "(no documents)"
        ans = read_ctx(ctx, q["text"], RAG_MODEL, RAG_URL, SYS_CTX)
        _emit(tag, rows, q, ans, extra={"ctx_syms": [by_cid[c][-1]["symbol"] for c in nodes]})
    return rows

def arm_graph_resolve(docs, queries, tag):
    """Structured (entity-graph) + RESOLVE: traverse each retrieved entity node's
    supersedes-chain to its terminal (latest-ingested) version and serve only that.
    SAME structure and retrieval as graph_accumulate; the only change is resolution
    along the version edge -> expected to stay current. This is the cell §5 predicts:
    structure that *encodes supersession* lands on the resolving side."""
    from retriever import make_retriever
    ret = make_retriever(); ret.add(docs)
    by_cid = _entity_graph(docs)
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        nodes = _graph_nodes(ret, q["text"], by_cid) or [q["concept_id"]]
        parts = [by_cid[c][-1]["_text"] for c in nodes]          # LATEST per entity
        ctx = "\n".join(parts) or "(no documents)"
        ans = read_ctx(ctx, q["text"], RAG_MODEL, RAG_URL, SYS_CTX)
        _emit(tag, rows, q, ans, extra={"ctx_syms": [by_cid[c][-1]["symbol"] for c in nodes]})
    return rows

def arm_wiki_karpathy(docs, queries, tag):
    # build pages: one per article (concept_id), current symbol overwritten by seq
    pages = {}
    for d in sorted(docs, key=lambda x: x["ingest_seq"]):
        p = pages.setdefault(d["concept_id"], {"concept": d["concept"], "current": None, "history": []})
        if p["current"] and p["current"] != d["symbol"]:
            p["history"].append(p["current"])
        p["current"] = d["symbol"]
    # page text = question + CURRENT answer only (overwrite kills stale)
    page_list = [{"cid": cid, "text": f"{p['concept']}\nCurrent answer: {p['current']}"}
                 for cid, p in pages.items()]
    # FTS5 keyword index over page text (NO embeddings), Karpathy-faithful navigation
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE pg USING fts5(body, tokenize='porter')")
    for pg in page_list:
        con.execute("INSERT INTO pg(rowid, body) VALUES (?,?)", (pg["cid"], pg["text"]))
    con.commit()
    txt = {pg["cid"]: pg["text"] for pg in page_list}
    def fts(query):
        terms = re.findall(r"[A-Za-z0-9]+", query)
        if not terms: return []
        m = " OR ".join(terms)
        try:
            cur = con.execute("SELECT rowid FROM pg WHERE pg MATCH ? ORDER BY bm25(pg) LIMIT ?", (m, K))
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
    rows = _resume(tag)
    for qi in range(len(rows), len(queries)):
        q = queries[qi]
        hits = fts(q["text"]) or [q["concept_id"]]               # fallback: own article
        ctx = "\n".join(txt[c] for c in hits)
        ans = read_ctx(ctx, q["text"], RAG_MODEL, RAG_URL, SYS_CTX)
        _emit(tag, rows, q, ans)
    return rows


ARMS = {"closed_book": arm_closed_book, "full_dump_cic": arm_full_dump_cic,
        "vector_rag": arm_vector_rag, "wiki_karpathy": arm_wiki_karpathy,
        "resolve_free": arm_resolve_free,
        "graph_accumulate": arm_graph_accumulate, "graph_resolve": arm_graph_resolve}


# ---------------------------------------------------------------- checkpoint/score
def _resume(tag):
    return real_core.hoh_load_done(f"pool_{tag}{TAGSFX}")

def _emit(tag, rows, q, ans, extra=None):
    sc = real_core.hoh_score(ans, q["gold"], q["deprecated_answers"])
    row = {"gold": q["gold"], "resp": ans[:80], **sc,
           "n_dep": len(q["deprecated_answers"])}
    if extra: row.update(extra)
    rows.append(row)
    real_core.hoh_append(f"pool_{tag}{TAGSFX}", row)
    if len(rows) % 25 == 0:
        s = real_core.hoh_summarize(rows)
        print(f"  [{tag}] {len(rows)}/{N} acc={s['accuracy']} SER={s['ser']}", flush=True)


def run_one(arm, docs, queries):
    rows = ARMS[arm](docs, queries, arm)
    s = real_core.hoh_summarize(rows)
    real_core.hoh_write_results(f"pool_{arm}{TAGSFX}", f"pooled-{arm}{TAGSFX}", rows,
                                extra={"n_pool_docs": len(docs), "k": K})
    return s


def main():
    docs, queries = build_pool()
    print(f"pooled corpus: {len(docs)} docs, {len(queries)} queries", flush=True)
    arms = list(ARMS) if ARM == "all" else [ARM]
    for a in arms:
        print(f"=== {a} ===", flush=True)
        s = run_one(a, docs, queries)
        print(f"  -> acc={s['accuracy']} SER={s['ser']} stale={s['n_stale']}/{s['n']}", flush=True)


if __name__ == "__main__":
    main()
