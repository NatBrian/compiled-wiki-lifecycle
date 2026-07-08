import json
import math
import os
import re
from collections import Counter

from .constants import (
    EMBEDDINGS_FILE,
    CHUNK_TARGET_CHARS,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    EMBEDDING_TOP_K,
)
from .utils import atomic_write, read_maybe


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = mag_a = mag_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        mag_a += a[i] * a[i]
        mag_b += b[i] * b[i]
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


def split_into_chunks(body: str) -> list[str]:
    paragraphs = _extract_paragraphs(body)
    if not paragraphs:
        return []
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        pieces = _split_oversized(para)
        for piece in pieces:
            candidate = (buffer + "\n\n" + piece).strip() if buffer else piece
            if len(candidate) > CHUNK_MAX_CHARS and buffer:
                chunks.append(buffer)
                buffer = piece
            else:
                buffer = candidate
    if buffer:
        chunks.append(buffer)
    if len(chunks) > 1 and len(chunks[-1]) < CHUNK_MIN_CHARS:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()
    return chunks


def _extract_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in raw if p.strip()]


def _split_oversized(text: str) -> list[str]:
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) > CHUNK_MAX_CHARS and buf:
            pieces.append(buf)
            buf = s
        else:
            buf = cand
    if buf:
        pieces.append(buf)
    return pieces


class EmbeddingStore:
    def __init__(self, root: str):
        self.root = root
        self.path = os.path.join(root, EMBEDDINGS_FILE)
        self.model = ""
        self.dimensions = 0
        self.entries: list[dict] = []
        self.chunks: list[dict] = []

    def load(self) -> bool:
        raw = read_maybe(self.path)
        if not raw:
            return False
        try:
            data = json.loads(raw)
            self.model = data.get("model", "")
            self.dimensions = data.get("dimensions", 0)
            self.entries = data.get("entries", [])
            self.chunks = data.get("chunks", [])
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def save(self) -> None:
        data = {
            "version": 1,
            "model": self.model,
            "dimensions": self.dimensions,
            "entries": self.entries,
            "chunks": self.chunks,
        }
        atomic_write(self.path, json.dumps(data, indent=2))

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def find_top_k_pages(self, query_vec: list[float], k: int = EMBEDDING_TOP_K) -> list[dict]:
        scored = [
            {"slug": e["slug"], "title": e["title"], "summary": e.get("summary", ""),
             "score": cosine_similarity(query_vec, e["vector"])}
            for e in self.entries
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def find_top_k_chunks(self, query_vec: list[float], k: int) -> list[dict]:
        scored = [
            {
                "slug": c["slug"],
                "title": c.get("title", ""),
                "chunkIndex": c.get("chunkIndex", 0),
                "text": c.get("text", ""),
                "score": cosine_similarity(query_vec, c["vector"]),
            }
            for c in self.chunks
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def update_page(self, slug: str, title: str, summary: str, body: str,
                    page_vec: list[float], chunk_vecs: list[tuple[str, list[float]]]) -> None:
        # Remove old entries for this slug
        self.entries = [e for e in self.entries if e["slug"] != slug]
        self.chunks = [c for c in self.chunks if c["slug"] != slug]

        self.entries.append({
            "slug": slug,
            "title": title,
            "summary": summary,
            "vector": page_vec,
        })

        for idx, (chunk_text, vec) in enumerate(chunk_vecs):
            self.chunks.append({
                "slug": slug,
                "title": title,
                "chunkIndex": idx,
                "text": chunk_text,
                "vector": vec,
            })

    def remove_page(self, slug: str) -> None:
        self.entries = [e for e in self.entries if e["slug"] != slug]
        self.chunks = [c for c in self.chunks if c["slug"] != slug]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               avg_dl: float, num_docs: int, doc_freqs: dict[str, int],
               k1: float = 1.5, b: float = 0.75) -> float:
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0
    score = 0.0
    doc_counter = Counter(doc_tokens)
    for qt in query_tokens:
        if qt not in doc_freqs or doc_freqs[qt] == 0:
            continue
        tf = doc_counter.get(qt, 0)
        idf = math.log(1 + (num_docs - doc_freqs[qt] + 0.5) / (doc_freqs[qt] + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


class BM25Ranker:
    def __init__(self, candidates: list[dict]):
        self.candidates = candidates
        self._build_index()

    def _build_index(self) -> None:
        self.num_docs = len(self.candidates)
        all_tokens: list[list[str]] = []
        for c in self.candidates:
            tokens = _tokenize(c.get("text", ""))
            all_tokens.append(tokens)
        self.all_tokens = all_tokens
        self.avg_dl = sum(len(t) for t in all_tokens) / max(self.num_docs, 1)
        self.doc_freqs: dict[str, int] = {}
        for tokens in all_tokens:
            seen = set(tokens)
            for t in seen:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

    def rerank(self, query: str) -> list[dict]:
        query_tokens = _tokenize(query)
        scored = []
        for i, c in enumerate(self.candidates):
            score = bm25_score(query_tokens, self.all_tokens[i],
                               self.avg_dl, self.num_docs, self.doc_freqs)
            scored.append({**c, "bm25_score": score})
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored


def embed_text(provider, text: str, input_type: str = "document") -> list[float]:
    return provider.embed(text, input_type)
