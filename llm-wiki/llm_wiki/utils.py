import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

import yaml

from .constants import (
    INDEX_FILE,
    LOG_FILE,
    LOCK_FILE,
    STATE_FILE,
    WIKI_DIR,
    CONCEPTS_DIR,
    LLMWIKI_DIR,
    QUERIES_DIR,
    LOG_MAX_PAGE_LINKS,
    RETRY_COUNT,
    RETRY_BASE_MS,
    RETRY_MULTIPLIER,
)
from .types import WikiState, Frontmatter, SourceState, IndexEntry


def slugify(text: str, max_len: int = 60) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dirs(root: str) -> None:
    for d in [WIKI_DIR, CONCEPTS_DIR, QUERIES_DIR, LLMWIKI_DIR]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def numbered_lines(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return ""
    width = len(str(len(lines)))
    return "\n".join(f"{str(i).rjust(width)} | {line}" for i, line in enumerate(lines, 1))


def build_concept_to_sources_map(state: WikiState) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for fname, s in state.sources.items():
        for slug in s.concepts:
            mapping.setdefault(slug, []).append(fname)
    return mapping


def format_source_sections(
    sources: list[tuple[str, str]], budget: int = 200_000
) -> str:
    if not sources:
        return ""
    total = sum(len(c) for _, c in sources)
    if total > budget:
        fair = budget // len(sources)
        adjusted = []
        for fname, content in sources:
            if len(content) > fair:
                adjusted.append(
                    (fname, content[:fair] + "\n[…truncated for prompt budget—see #39…]")
                )
            else:
                adjusted.append((fname, content))
        sources = adjusted
    sections = []
    for fname, content in sources:
        sections.append(f"--- SOURCE: {fname} ---\n{numbered_lines(content)}")
    return "\n\n".join(sections)


def parse_frontmatter(text: str) -> tuple[Frontmatter, str]:
    fm = Frontmatter()
    body = text.strip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            yaml_text = parts[1].strip()
            body = parts[2].strip()
            try:
                data = yaml.safe_load(yaml_text)
                if isinstance(data, dict):
                    fm.title = data.get("title", "")
                    fm.summary = data.get("summary", "")
                    fm.sources = data.get("sources", [])
                    fm.kind = data.get("kind", "concept")
                    fm.createdAt = data.get("createdAt", "")
                    fm.updatedAt = data.get("updatedAt", "")
                    fm.confidence = data.get("confidence", 1.0)
                    fm.provenanceState = data.get("provenanceState", "extracted")
                    contradicted = data.get("contradictedBy", [])
                    if isinstance(contradicted, list):
                        fm.contradictedBy = [
                            {"slug": c.get("slug", ""), "reason": c.get("reason", "")}
                            if isinstance(c, dict)
                            else {"slug": str(c), "reason": ""}
                            for c in contradicted
                        ]
                    fm.tags = data.get("tags", [])
                    fm.orphaned = data.get("orphaned", False)
            except yaml.YAMLError:
                pass
    return fm, body


def format_frontmatter(fm: Frontmatter) -> str:
    data = {}
    for key in ("title", "summary", "kind", "createdAt", "updatedAt", "confidence",
                 "provenanceState", "orphaned", "tags"):
        val = getattr(fm, key, None)
        if val is not None and val != "" and val != [] and val != 0.0 and val is not False:
            data[key] = val
    if fm.sources:
        data["sources"] = fm.sources
    if fm.contradictedBy:
        data["contradictedBy"] = [
            {"slug": c["slug"], "reason": c["reason"]}
            for c in fm.contradictedBy
        ]
    docs = []
    if fm.createdAt:
        docs.append(f"createdAt: {fm.createdAt}")
    if fm.updatedAt:
        docs.append(f"updatedAt: {fm.updatedAt}")
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yaml_str}\n---\n"


def atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def read_maybe(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_state(root: str) -> WikiState:
    path = os.path.join(root, STATE_FILE)
    raw = read_maybe(path)
    if not raw:
        return WikiState()
    try:
        data = json.loads(raw)
        state = WikiState(version=data.get("version", 1), indexHash=data.get("indexHash", ""))
        for fname, s in data.get("sources", {}).items():
            state.sources[fname] = SourceState(
                hash=s.get("hash", ""),
                concepts=s.get("concepts", []),
                compiledAt=s.get("compiledAt", ""),
            )
        state.frozenSlugs = data.get("frozenSlugs", [])
        return state
    except (json.JSONDecodeError, TypeError):
        bak_path = path + ".bak"
        try:
            shutil.copy2(path, bak_path)
        except Exception:
            pass
        return WikiState()


def save_state(root: str, state: WikiState) -> None:
    data = {
        "version": state.version,
        "sources": {
            fname: {"hash": s.hash, "concepts": s.concepts, "compiledAt": s.compiledAt}
            for fname, s in state.sources.items()
        },
        "frozenSlugs": state.frozenSlugs,
        "indexHash": state.indexHash,
    }
    path = os.path.join(root, STATE_FILE)
    atomic_write(path, json.dumps(data, indent=2))


def acquire_lock(root: str, timeout_sec: float = 30.0) -> bool:
    lock_file = os.path.join(root, LOCK_FILE)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.5)
    return False


def release_lock(root: str) -> None:
    lock_file = os.path.join(root, LOCK_FILE)
    try:
        os.unlink(lock_file)
    except FileNotFoundError:
        pass


def append_log(root: str, entry: dict) -> None:
    log_file = os.path.join(root, LOG_FILE)
    now = now_iso()
    date_str = now[:10]
    action = entry.get("action", "unknown")
    source_count = entry.get("sourceCount", 0)
    page_count = entry.get("pageCount", 0)
    duration = entry.get("durationMs", 0)
    created = entry.get("created", [])
    updated = entry.get("updated", [])
    deleted_sources = entry.get("deletedSources", [])
    sources = entry.get("sources", [])

    lines = [
        f"## [{date_str}] {action} | {source_count} source(s) \u2192 {page_count} page(s) ({duration}ms)"
    ]

    if sources:
        parts = []
        for s in sources[:LOG_MAX_PAGE_LINKS]:
            label = s[:200]
            parts.append(label)
        if parts:
            lines.append(f"- Sources: {', '.join(parts)}")

    if created:
        created_links = [f"[[{s}]]" for s in created[:LOG_MAX_PAGE_LINKS]]
        if created_links:
            lines.append(f"- Created: {', '.join(created_links)}")

    if updated:
        updated_links = [f"[[{s}]]" for s in updated[:LOG_MAX_PAGE_LINKS]]
        if updated_links:
            lines.append(f"- Updated: {', '.join(updated_links)}")

    if deleted_sources:
        lines.append(f"- Deleted sources: {len(deleted_sources)}")

    entry_text = "\n".join(lines) + "\n\n"
    old = read_maybe(log_file)
    with open(log_file, "w") as f:
        f.write((old + entry_text) if old else ("# LLM-Wiki Log\n\n" + entry_text))


def generate_index(root: str, source_count: int, page_count: int,
                   duration_ms: int, query_results: Optional[list] = None) -> str:
    content = "# Wiki Index\n\n"
    pages_dir = os.path.join(root, CONCEPTS_DIR)
    pages = []
    if os.path.isdir(pages_dir):
        for fname in sorted(os.listdir(pages_dir)):
            if fname.endswith(".md"):
                slug = fname[:-3]
                page_path = os.path.join(pages_dir, fname)
                raw = read_maybe(page_path)
                fm, body = parse_frontmatter(raw)
                if fm.orphaned:
                    continue
                summary = fm.summary or (body[:120].strip() + "..." if len(body) > 120 else body.strip())
                pages.append(IndexEntry(slug=slug, title=fm.title or slug, summary=summary))

    for p in pages:
        content += f"- **[[{p.slug}|{p.title}]]**, {p.summary}\n"

    content += "\n## Summary\n\n"
    content += f"- **Sources:** {source_count}\n"
    content += f"- **Pages:** {page_count}\n"
    content += f"- **Last compiled:** {now_iso()}\n"
    content += f"- **Duration:** {duration_ms}ms\n"

    return content


def select_related(entries: list[IndexEntry], current_slug: str, max_count: int) -> list[str]:
    slugs = [e.slug for e in entries if e.slug != current_slug]
    return slugs[:max_count]


def with_retry(fn, retries: int = RETRY_COUNT, base_ms: int = RETRY_BASE_MS):
    last_ex = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as ex:
            last_ex = ex
            if attempt < retries - 1:
                delay = base_ms * (RETRY_MULTIPLIER ** attempt) / 1000
                time.sleep(delay)
    raise last_ex
