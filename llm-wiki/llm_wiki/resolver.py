import os
import re

from .types import Frontmatter
from .utils import read_maybe, parse_frontmatter, format_frontmatter, atomic_write


# ── Guard functions ────────────────────────────────────────────────

def _is_inside_wikilink(text: str, pos: int) -> bool:
    before = text.rfind("[[", 0, pos)
    if before == -1:
        return False
    after = text.find("]]", before)
    return after >= pos


def _is_inside_citation(text: str, pos: int) -> bool:
    before = text.rfind("^[", 0, pos)
    if before == -1:
        return False
    after = text.find("]", before)
    return after >= pos


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    boundary_before = start == 0 or bool(
        re.match(r"[\s,.:;!?()\[\]{}/\"']", text[start - 1])
    )
    boundary_after = end >= len(text) or bool(
        re.match(r"[\s,.:;!?()\[\]{}/\"']", text[end])
    )
    return boundary_before and boundary_after


# ── Title index ────────────────────────────────────────────────────

def build_title_index(root: str, concepts_dir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    if not os.path.isdir(concepts_dir):
        return index
    for fname in sorted(os.listdir(concepts_dir)):
        if not fname.endswith(".md"):
            continue
        slug = fname[:-3]
        raw = read_maybe(os.path.join(concepts_dir, fname))
        if not raw:
            continue
        fm, _ = parse_frontmatter(raw)
        if fm.orphaned:
            continue
        title = fm.title or slug
        index[slug] = title
    return index


# ── Outbound link resolution ───────────────────────────────────────

def _resolve_outbound(body: str, page_slug: str, title_index: dict[str, str]) -> str:
    all_replacements: list[tuple[int, int, str]] = []

    for title, target_slug in sorted(title_index.items(), key=lambda x: -len(x[0])):
        if target_slug == page_slug:
            continue
        pattern = re.compile(re.escape(title))
        for m in pattern.finditer(body):
            start, end = m.start(), m.end()
            if not _is_word_boundary(body, start, end):
                continue
            if _is_inside_wikilink(body, start) or _is_inside_wikilink(body, end - 1):
                continue
            if _is_inside_citation(body, start) or _is_inside_citation(body, end - 1):
                continue
            all_replacements.append((start, end, f"[[{target_slug}|{title}]]"))

    if not all_replacements:
        return body

    all_replacements.sort(key=lambda x: x[0], reverse=True)
    result = body
    for start, end, replacement in all_replacements:
        result = result[:start] + replacement + result[end:]
    return result


# ── Inbound link resolution ────────────────────────────────────────

def _resolve_inbound(body: str, new_title: str, new_slug: str) -> str:
    replacements: list[tuple[int, int, str]] = []
    pattern = re.compile(re.escape(new_title))
    for m in pattern.finditer(body):
        start, end = m.start(), m.end()
        if not _is_word_boundary(body, start, end):
            continue
        if _is_inside_wikilink(body, start) or _is_inside_wikilink(body, end - 1):
            continue
        if _is_inside_citation(body, start) or _is_inside_citation(body, end - 1):
            continue
        replacements.append((start, end, f"[[{new_slug}|{new_title}]]"))

    if not replacements:
        return body

    replacements.sort(key=lambda x: x[0], reverse=True)
    result = body
    for start, end, replacement in replacements:
        result = result[:start] + replacement + result[end:]
    return result


# ── Two-pass resolution ────────────────────────────────────────────

def resolve_links(
    root: str,
    concepts_dir: str,
    changed_slugs: list[str],
    new_slugs: list[str],
) -> None:
    title_index = build_title_index(root, concepts_dir)
    if not title_index:
        return

    # Pass 1: Outbound on changed pages
    for slug in changed_slugs:
        page_path = os.path.join(concepts_dir, f"{slug}.md")
        raw = read_maybe(page_path)
        if not raw:
            continue
        fm, body = parse_frontmatter(raw)
        resolved_body = _resolve_outbound(body, slug, title_index)
        if resolved_body == body:
            continue
        new_content = format_frontmatter(fm) + resolved_body
        if not new_content.endswith("\n"):
            new_content += "\n"
        atomic_write(page_path, new_content)

    # Pass 2: Inbound on ALL pages for new titles
    for slug in new_slugs:
        new_title = title_index.get(slug)
        if not new_title:
            continue
        for fname in sorted(os.listdir(concepts_dir)):
            if not fname.endswith(".md"):
                continue
            page_slug = fname[:-3]
            if page_slug == slug or page_slug in changed_slugs:
                continue
            page_path = os.path.join(concepts_dir, fname)
            raw = read_maybe(page_path)
            if not raw:
                continue
            fm, body = parse_frontmatter(raw)
            resolved_body = _resolve_inbound(body, new_title, slug)
            if resolved_body == body:
                continue
            new_content = format_frontmatter(fm) + resolved_body
            if not new_content.endswith("\n"):
                new_content += "\n"
            atomic_write(page_path, new_content)


# ── Existing resolver helpers (preserved) ──────────────────────────

def resolve_wikilinks(text: str, slug: str, known_slugs: set[str]) -> str:
    def repl_with(m: re.Match) -> str:
        target = m.group(1)
        display = m.group(2)
        if target in known_slugs:
            return f"[[{target}|{display}]]"
        return display

    def repl_bare(m: re.Match) -> str:
        target = m.group(1)
        if target in known_slugs:
            return f"[[{target}|{target}]]"
        return target

    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", repl_with, text)
    text = re.sub(r"\[\[([^\]]+)\]\]", repl_bare, text)
    return text


def get_all_pages(root: str, concepts_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(concepts_dir):
        return []
    result = []
    for fname in sorted(os.listdir(concepts_dir)):
        if fname.endswith(".md"):
            slug = fname[:-3]
            content = read_maybe(os.path.join(concepts_dir, fname))
            result.append((slug, content))
    return result


def get_incoming_links(all_pages: list[tuple[str, str]], target_slug: str) -> set[str]:
    incoming = set()
    for slug, content in all_pages:
        if slug == target_slug:
            continue
        pat1 = re.compile(r"\[\[" + re.escape(target_slug) + r"\|")
        pat2 = re.compile(r"\[\[" + re.escape(target_slug) + r"\]\]")
        if pat1.search(content) or pat2.search(content):
            incoming.add(slug)
    return incoming


def update_orphan_status(root: str, concepts_dir: str) -> None:
    all_pages = get_all_pages(root, concepts_dir)
    for slug, content in all_pages:
        page_path = os.path.join(concepts_dir, f"{slug}.md")
        fm, body = parse_frontmatter(content)
        incoming = get_incoming_links(all_pages, slug)
        should_be_orphaned = len(incoming) == 0
        if fm.orphaned != should_be_orphaned:
            fm.orphaned = should_be_orphaned
            new_content = format_frontmatter(fm) + body
            if not new_content.endswith("\n"):
                new_content += "\n"
            atomic_write(page_path, new_content)


def orphan_page(root: str, slug: str, concepts_dir: str) -> None:
    page_path = os.path.join(concepts_dir, f"{slug}.md")
    raw = read_maybe(page_path)
    if not raw:
        return
    fm, _ = parse_frontmatter(raw)
    if fm.orphaned:
        return
    # Surgical insert: replace first "---\n" with "---\norphaned: true\n"
    updated = raw.replace("---\n", "---\norphaned: true\n", 1)
    atomic_write(page_path, updated)
