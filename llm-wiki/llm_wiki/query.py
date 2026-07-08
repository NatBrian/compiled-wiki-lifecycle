import json
import os
import re
import time
from datetime import datetime, timezone

from .types import QueryResult
from .constants import (
    CONCEPTS_DIR,
    QUERIES_DIR,
    INDEX_FILE,
    QUERY_PAGE_LIMIT,
    CHUNK_TOP_K,
    CHUNK_RERANK_KEEP,
    EMBEDDING_TOP_K,
)
from .utils import atomic_write, read_maybe, parse_frontmatter, now_iso, ensure_dirs, append_log
from .llm import get_provider, LLMError
from .prompts import (
    QUERY_SELECTION_SYSTEM,
    QUERY_SELECTION_PROMPT,
    QUERY_ANSWER_SYSTEM,
    QUERY_ANSWER_PROMPT,
)
from .embeddings import EmbeddingStore, split_into_chunks, BM25Ranker

# How many additional linked pages to pull in via wikilink graph expansion
QUERY_WIKILINK_EXPAND_LIMIT = 3

PAGE_DIRS = [CONCEPTS_DIR, QUERIES_DIR]


def _load_wiki_pages(concepts_dir: str) -> list[dict]:
    pages = []
    if not os.path.isdir(concepts_dir):
        return pages
    for fname in sorted(os.listdir(concepts_dir)):
        if fname.endswith(".md"):
            slug = fname[:-3]
            raw = read_maybe(os.path.join(concepts_dir, fname))
            fm, body = parse_frontmatter(raw)
            if fm.orphaned:
                continue
            pages.append({
                "slug": slug,
                "title": fm.title or slug,
                "summary": fm.summary or body[:200],
                "content": body,
            })
    return pages


def _load_index(root: str) -> str:
    return read_maybe(os.path.join(root, INDEX_FILE))


def _select_pages_via_llm(provider, question: str, index_content: str) -> list[str]:
    prompt = QUERY_SELECTION_PROMPT.format(question=question, indexContent=index_content)
    try:
        slugs = provider.select_pages(QUERY_SELECTION_SYSTEM, prompt, max_items=QUERY_PAGE_LIMIT)
        if slugs and isinstance(slugs, list):
            return slugs[:QUERY_PAGE_LIMIT]
    except (NotImplementedError, LLMError):
        pass
    # Fallback: use completion
    try:
        raw = provider.complete(QUERY_SELECTION_SYSTEM, prompt)
        parsed = json.loads(raw)
        slugs = parsed.get("selected", [])
        if isinstance(slugs, list) and len(slugs) > 0:
            return [s for s in slugs if isinstance(s, str)][:QUERY_PAGE_LIMIT]
    except (json.JSONDecodeError, TypeError, LLMError):
        pass
    return []


def _expand_wikilinks(root: str, slugs: list[str], max_expand: int = QUERY_WIKILINK_EXPAND_LIMIT) -> list[str]:
    expanded = list(slugs)
    seen = set(slugs)
    concepts_dir = os.path.join(root, CONCEPTS_DIR)
    wikilink_re = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]")
    for slug in slugs:
        raw = read_maybe(os.path.join(concepts_dir, f"{slug}.md"))
        if not raw:
            continue
        fm, body = parse_frontmatter(raw)
        if fm.orphaned:
            continue
        for m in wikilink_re.finditer(body):
            target = m.group(1)
            if target in seen:
                continue
            target_path = os.path.join(concepts_dir, f"{target}.md")
            target_raw = read_maybe(target_path)
            if not target_raw:
                continue
            target_fm, _ = parse_frontmatter(target_raw)
            if target_fm.orphaned:
                continue
            seen.add(target)
            expanded.append(target)
            if len(expanded) >= len(slugs) + max_expand:
                break
        if len(expanded) >= len(slugs) + max_expand:
            break
    return expanded


def _load_selected_pages(root: str, slugs: list[str]) -> str:
    sections = []
    for slug in slugs:
        content = ""
        for d in PAGE_DIRS:
            raw = read_maybe(os.path.join(root, d, f"{slug}.md"))
            if raw:
                fm, body = parse_frontmatter(raw)
                if fm.orphaned:
                    continue
                content = body
                break
        if content:
            sections.append(f"--- Page: {slug} ---\n{content}")
    return "\n\n".join(sections)


def query(root: str, question: str, save: bool = False) -> QueryResult:
    result = QueryResult(question=question)
    t0 = time.monotonic()

    ensure_dirs(root)
    concepts_dir = os.path.join(root, CONCEPTS_DIR)
    pages = _load_wiki_pages(concepts_dir)

    if not pages:
        result.answer = "No wiki pages available yet."
        return result

    provider = get_provider()
    selected_slugs: list[str] = []

    # --- Tier 1: Chunk-level embedding retrieval ---
    if not selected_slugs:
        try:
            store = EmbeddingStore(root)
            if store.load() and not store.is_empty():
                query_vec = provider.embed(question, input_type="query")
                chunk_hits = store.find_top_k_chunks(query_vec, CHUNK_TOP_K)
                if chunk_hits:
                    ranker = BM25Ranker(chunk_hits)
                    reranked = ranker.rerank(question)
                    kept = reranked[:CHUNK_RERANK_KEEP]
                    seen = set()
                    for c in kept:
                        if c["slug"] not in seen:
                            selected_slugs.append(c["slug"])
                            seen.add(c["slug"])
                        if len(selected_slugs) >= QUERY_PAGE_LIMIT:
                            break
                    print(f"  [query] chunk retrieval: {len(chunk_hits)} chunks → {len(selected_slugs)} pages")
        except (LLMError, NotImplementedError) as e:
            print(f"  [query] chunk retrieval unavailable ({e}); falling back")

    # --- Tier 2: Page-level embedding retrieval ---
    if not selected_slugs:
        try:
            store = EmbeddingStore(root)
            if store.load() and not store.is_empty():
                query_vec = provider.embed(question, input_type="query")
                page_hits = store.find_top_k_pages(query_vec, EMBEDDING_TOP_K)
                selected_slugs = [p["slug"] for p in page_hits[:QUERY_PAGE_LIMIT]]
                print(f"  [query] page retrieval: {len(page_hits)} candidates → {len(selected_slugs)} pages")
        except (LLMError, NotImplementedError) as e:
            print(f"  [query] page retrieval unavailable ({e}); falling back")

    # --- Tier 3: LLM reads index.md ---
    if not selected_slugs:
        index_content = _load_index(root)
        if index_content:
            selected_slugs = _select_pages_via_llm(provider, question, index_content)
            print(f"  [query] LLM index selection: {len(selected_slugs)} pages")
        else:
            selected_slugs = [p["slug"] for p in pages[:3]]

    # Expand selected pages via wikilink graph (follow [[links]] one level deep)
    expanded_slugs = _expand_wikilinks(root, selected_slugs)
    if len(expanded_slugs) > len(selected_slugs):
        added = len(expanded_slugs) - len(selected_slugs)
        print(f"  [query] wikilink expansion: +{added} linked pages")
        selected_slugs = expanded_slugs

    result.selectedPages = selected_slugs

    pages_content = _load_selected_pages(root, selected_slugs)
    if not pages_content:
        result.answer = "No matching pages found."
        return result

    answer_prompt = QUERY_ANSWER_PROMPT.format(question=question, pages=pages_content)
    try:
        result.answer = provider.complete(QUERY_ANSWER_SYSTEM, answer_prompt)
    except LLMError as e:
        result.answer = f"Answer generation failed: {e}"

    if save:
        queries_dir = os.path.join(root, QUERIES_DIR)
        os.makedirs(queries_dir, exist_ok=True)
        safe_name = question.lower().replace(" ", "-")[:40].rstrip("-")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"query-{safe_name}-{ts}.md"
        content = f"# Query: {question}\n\n"
        content += f"**Date:** {now_iso()}\n\n"
        content += "## Selected Pages\n\n"
        for slug in selected_slugs:
            content += f"- [[{slug}]]\n"
        content += "\n## Answer\n\n"
        content += result.answer + "\n"
        atomic_write(os.path.join(queries_dir, fname), content)
        result.saved = True

        # Regenerate index so saved query is discoverable
        from .compiler import compile_wiki
        # Lightweight index refresh only
        from .utils import generate_index
        page_count = len(os.listdir(concepts_dir)) if os.path.isdir(concepts_dir) else 0
        idx = generate_index(root, 0, page_count, int((time.monotonic() - t0) * 1000))
        atomic_write(os.path.join(root, INDEX_FILE), idx)

    # Log the query
    append_log(root, {
        "action": "query",
        "sourceCount": 0,
        "pageCount": len(selected_slugs),
        "durationMs": int((time.monotonic() - t0) * 1000),
        "created": [],
        "updated": [],
        "sources": [],
        "deletedSources": [],
    })

    return result
