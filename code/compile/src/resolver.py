"""Label-free supersession resolver (the deployable lever, P2).

The A5 SupersededRAG arm collapsed retrieved chunks to the latest version per
concept using the GOLD concept_id (`id.split("__")[0]`) and the GOLD
`version_introduced` field. That is an oracle: an arbitrary corpus does not hand
you a concept id or a clean version tag.

This module removes both oracle signals. It groups retrieved chunks purely by
the SEMANTIC SIMILARITY of their observable text (any real document has prose),
and orders them purely by INGESTION ORDER (a system always knows when it added a
chunk -- this is fair, not oracle). Within each semantic cluster it keeps the
latest-ingested chunk and optionally surfaces the older forms as explicit history.

resolve_chunks() is a free function so it can be bolted on top of ANY retriever's
output (our arms, or a real published system's retrieved context) -- this is the
composability claim (P1).
"""
from __future__ import annotations

import os

_THRESH = float(os.environ.get("RESOLVE_THRESH", "0.86"))
_MODEL = os.environ.get("RESOLVE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

_embedder = None


def _embed(texts: list[str]):
    """Encode texts to L2-normalized vectors on CPU (no GPU contention)."""
    global _embedder
    import numpy as np
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL, device="cpu")
    emb = _embedder.encode(texts, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def cluster_by_similarity(texts: list[str], thresh: float = _THRESH):
    """Greedy single-pass clustering by cosine similarity of embeddings.

    Returns a list of clusters, each a list of indices into `texts`. A chunk
    joins the most-similar existing cluster whose centroid similarity >= thresh,
    else it seeds a new cluster. Order-stable: clusters are seeded in input
    (retrieval-rank) order.
    """
    if not texts:
        return []
    import numpy as np
    embs = _embed(texts)
    centroids = []   # running sum vectors (un-normalized)
    members = []     # list[list[int]]
    for i, e in enumerate(embs):
        best_s, best_j = thresh, -1
        for j, cen in enumerate(centroids):
            c = cen / (np.linalg.norm(cen) + 1e-9)
            s = float(e @ c)
            if s >= best_s:
                best_s, best_j = s, j
        if best_j == -1:
            centroids.append(e.copy())
            members.append([i])
        else:
            centroids[best_j] += e
            members[best_j].append(i)
    return members


def resolve_chunks(chunks: list[dict], *, text_key: str = "_text",
                   seq_key: str = "ingest_seq", thresh: float = _THRESH,
                   keep_history: bool = False, symbol_key: str = "symbol"):
    """Collapse retrieved chunks to the latest-ingested chunk per semantic cluster.

    - chunks: retrieved docs, each a dict with `text_key` and `seq_key`.
    - keep_history: if True, attach `_hist` = " (was: <old symbols>)" so the
      old->new direction is recoverable for synthesis questions.
    Returns the kept chunks, in the retrieval rank order of their cluster's first
    appearance (preserves relevance ordering).
    """
    if not chunks:
        return []
    texts = [c.get(text_key, "") for c in chunks]
    clusters = cluster_by_similarity(texts, thresh)
    kept = []
    for members in clusters:
        latest = max(members, key=lambda i: chunks[i].get(seq_key, i))
        ch = dict(chunks[latest])
        if keep_history:
            olders = [chunks[i] for i in members if i != latest]
            olders = sorted(olders, key=lambda c: c.get(seq_key, 0))
            old_syms = [o.get(symbol_key, "") for o in olders if o.get(symbol_key)]
            ch["_hist"] = (" (was: " + ", ".join(old_syms) + ")") if old_syms else ""
        kept.append((min(members), ch))   # rank = earliest member position
    kept.sort(key=lambda t: t[0])
    return [c for _, c in kept]
