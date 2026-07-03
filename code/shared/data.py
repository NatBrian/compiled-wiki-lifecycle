"""Load SciFact-Open corpus + claims; subsample corpus to size N keeping gold docs.

Formats (from scifact-open/doc/data.md):
  corpus.jsonl : {"doc_id": int, "title": str, "abstract": [sent, ...], ...}
  claims.jsonl : {"id": int, "claim": str,
                  "evidence": {"<doc_id>": {"label": "SUPPORT"|"CONTRADICT",
                                            "sentences": [...], ...}, ...}}
NEI is implicit: any (claim, doc) not in evidence => NEI.
"""
import json
import random


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_corpus(corpus_path):
    docs = {}
    for d in load_jsonl(corpus_path):
        docs[int(d["doc_id"])] = {
            "doc_id": int(d["doc_id"]),
            "title": d.get("title", ""),
            "text": " ".join(d["abstract"]) if isinstance(d.get("abstract"), list) else d.get("abstract", ""),
        }
    return docs


def load_claims(claims_path):
    claims = []
    for c in load_jsonl(claims_path):
        ev = {int(k): v for k, v in c.get("evidence", {}).items()}
        claims.append({"id": c["id"], "claim": c["claim"], "evidence": ev})
    return claims


def gold_doc_ids(claims):
    ids = set()
    for c in claims:
        ids.update(c["evidence"].keys())
    return ids


def subsample_corpus(docs, claims, N, seed=0):
    """Return a corpus subset of size N that ALWAYS contains every gold evidence doc.
    This is required: dropping a gold doc would make 'NEI' the correct answer and
    destroy the false-absence measurement."""
    gold = gold_doc_ids(claims) & set(docs.keys())
    if N < len(gold):
        raise ValueError(f"N={N} < #gold={len(gold)}; raise N to keep all gold docs")
    rng = random.Random(seed)
    pool = [i for i in docs if i not in gold]
    n_fill = N - len(gold)
    chosen = set(gold) | set(rng.sample(pool, min(n_fill, len(pool))))
    return {i: docs[i] for i in chosen}


def mini_corpus_for_claims(docs, claim_subset, N, seed=0):
    """Build a small corpus of size N containing the gold docs of ONLY the given claims
    (plus random fill). For cheap CPU smoke tests so compile doesn't process all 406 gold."""
    gold = set()
    for c in claim_subset:
        gold |= {d for d in c["evidence"] if d in docs}
    if N < len(gold):
        N = len(gold)
    rng = random.Random(seed)
    pool = [i for i in docs if i not in gold]
    chosen = set(gold) | set(rng.sample(pool, min(N - len(gold), len(pool))))
    return {i: docs[i] for i in chosen}


def supported_claims(claims, corpus_ids):
    """Claims that DO have an in-corpus SUPPORT/CONTRADICT doc. For these the correct
    answer is NOT NEI, so a NEI prediction = false-absence (our headline metric)."""
    out = []
    for c in claims:
        hits = {d: v for d, v in c["evidence"].items()
                if d in corpus_ids and v["label"] in ("SUPPORT", "CONTRADICT")}
        if hits:
            out.append({**c, "in_corpus_evidence": hits})
    return out
