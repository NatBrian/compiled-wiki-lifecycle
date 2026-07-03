"""LOFT accuracy-at-scale control (NO staleness — complements pooled-HoH).

LOFT (DeepMind, arXiv 2406.13121) Corpus-in-Context: ID-tagged corpus at fixed
token budgets (32k/128k). Standard retrieval QA, no superseded facts. Used here to
test the ACCURACY/SCALE axis the user asked for ("limit/accuracy/performance of
llm-wiki, rag, full-context dump") that pooled-HoH (staleness/scale) does not cover:

  full_dump_cic : LOFT-native CiC, whole corpus ID-tagged in context (256K reader).
                  At 128k (~80-90k tok) this is a true long-context stress.
  vector_rag    : dense bge top-k retrieval over the same corpus (14B reader).

Score: subspan match — any gold answer string appearing in the response (LOFT's
relaxed EM). Datasets nq, hotpotqa at 32k, 128k.

Run: real/venvs/lightrag/bin/python real/loft_run.py <arm> <dataset> <scale>
     arm in {full_dump_cic, vector_rag}; dataset in {nq, hotpotqa}; scale {32k,128k}
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
LOFT = ROOT / "loft_data"

DUMP_MODEL = os.environ.get("DUMP_MODEL", "qwen3-coder-30b")
DUMP_URL = os.environ.get("DUMP_URL", "http://127.0.0.1:8102/v1")
RAG_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-14b")
RAG_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8101/v1")
K = int(os.environ.get("LOFT_K", "5"))

_clients = {}
def _client(url):
    if url not in _clients:
        _clients[url] = OpenAI(base_url=url, api_key="dummy")
    return _clients[url]

def chat(model, url, system, user, max_tokens=64):
    r = _client(url).chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


def load(dataset, scale):
    base = LOFT / dataset / scale
    corpus = [json.loads(l) for l in (base / "corpus.jsonl").read_text().splitlines() if l.strip()]
    queries = [json.loads(l) for l in (base / "dev_queries.jsonl").read_text().splitlines() if l.strip()]
    for d in corpus:
        d["_text"] = (d.get("title_text", "") + " " + d.get("passage_text", "")).strip()
    return corpus, queries


def _match(resp, answers):
    r = re.sub(r"\s+", " ", resp.lower())
    return any(re.sub(r"\s+", " ", a.lower()) in r for a in answers)


SYS_CIC = ("You are given a CORPUS of ID-tagged documents. Use ONLY the corpus to "
           "answer the question. Reply with just the answer, as short as possible.")
SYS_RAG = ("Use ONLY the context to answer the question. Reply with just the answer, "
           "as short as possible.")


def run(arm, dataset, scale):
    corpus, queries = load(dataset, scale)
    rows = []
    if arm == "full_dump_cic":
        cic = "\n".join(f"[{d['pid']}] {d['_text']}" for d in corpus)
        for q in queries:
            ans = chat(DUMP_MODEL, DUMP_URL, SYS_CIC,
                       f"Corpus:\n{cic}\n\nQuestion: {q['query_text']}")
            rows.append({"q": q["query_text"], "gold": q["answers"], "resp": ans[:80],
                         "correct": _match(ans, q["answers"])})
    elif arm == "vector_rag":
        from retriever import make_retriever
        ret = make_retriever(); ret.add(corpus)
        for q in queries:
            hits = ret.search(q["query_text"], K)
            ctx = "\n".join(h["_text"] for h in hits)
            ans = chat(RAG_MODEL, RAG_URL, SYS_RAG,
                       f"Context:\n{ctx}\n\nQuestion: {q['query_text']}")
            rows.append({"q": q["query_text"], "gold": q["answers"], "resp": ans[:80],
                         "correct": _match(ans, q["answers"])})
    else:
        raise SystemExit(f"unknown arm {arm}")
    n = len(rows); acc = round(sum(r["correct"] for r in rows) / n, 3) if n else 0
    out = {"arm": arm, "dataset": dataset, "scale": scale, "n": n, "accuracy": acc,
           "n_corpus": len(corpus), "rows": rows}
    tag = f"loft_{arm}_{dataset}_{scale}"
    (ROOT / "results" / f"results_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[loft] {arm} {dataset} {scale}: acc={acc} (n={n}, corpus={len(corpus)})", flush=True)
    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
