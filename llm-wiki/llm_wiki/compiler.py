import os
import time

from .types import (
    ChangeType,
    SourceChange,
    CompileResult,
    Frontmatter,
    ExtractedConcept,
    ExtractedConceptWithSource,
    MergedConcept,
    SourceState,
    Contradiction,
)
from .constants import (
    SOURCES_DIR,
    CONCEPTS_DIR,
    MIN_SOURCE_CHARS,
    MAX_SOURCE_CHARS,
    resolve_prompt_budget_chars,
    MAX_CONCEPTS,
    DEFAULT_COMPILE_CONCURRENCY,
    INDEX_FILE,
)
from .utils import (
    sha256,
    ensure_dirs,
    now_iso,
    parse_frontmatter,
    format_frontmatter,
    atomic_write,
    read_maybe,
    load_state,
    save_state,
    acquire_lock,
    release_lock,
    append_log,
    generate_index,
    select_related,
    slugify,
    numbered_lines,
    build_concept_to_sources_map,
    format_source_sections,
)
from .llm import get_provider, LLMError
from .prompts import EXTRACTION_SYSTEM, EXTRACTION_PROMPT, PAGE_GEN_SYSTEM, PAGE_GEN_PROMPT
from .resolver import (
    get_all_pages,
    update_orphan_status,
    resolve_wikilinks,
    build_title_index,
    resolve_links,
    orphan_page,
)
from .embeddings import EmbeddingStore, split_into_chunks


def _load_sources(root: str) -> list[tuple[str, str]]:
    sources_dir = os.path.join(root, SOURCES_DIR)
    if not os.path.isdir(sources_dir):
        return []
    result = []
    for fname in sorted(os.listdir(sources_dir)):
        fpath = os.path.join(sources_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".md"):
            content = read_maybe(fpath)
            if len(content) >= MIN_SOURCE_CHARS:
                if len(content) > MAX_SOURCE_CHARS:
                    print(f"  [warn] {fname} exceeds {MAX_SOURCE_CHARS} chars ({len(content)})")
                result.append((fname, content))
    return result


def _detect_changes(state, sources):
    known = set(state.sources.keys())
    current = set(fname for fname, _ in sources)
    source_map = dict(sources)

    changes = []
    deleted_slugs = set()
    for fname in sorted(current):
        content = source_map[fname]
        h = sha256(content)
        if fname not in state.sources:
            changes.append(SourceChange(file=fname, type=ChangeType.NEW))
        elif state.sources[fname].hash != h:
            changes.append(SourceChange(file=fname, type=ChangeType.CHANGED))
        elif state.sources[fname].hash == "":
            # Retry previously failed extraction
            changes.append(SourceChange(file=fname, type=ChangeType.CHANGED))

    for fname in sorted(known):
        if fname not in current:
            changes.append(SourceChange(file=fname, type=ChangeType.DELETED))
            if fname in state.sources:
                deleted_slugs.update(state.sources[fname].concepts)

    return changes, source_map, deleted_slugs


def _find_affected_sources(state, direct_changes, concept_to_sources):
    affected = set()
    for change in direct_changes:
        fname = change.file
        if fname not in state.sources:
            continue
        for slug in state.sources[fname].concepts:
            if slug not in concept_to_sources:
                continue
            for other in concept_to_sources[slug]:
                if other != fname:
                    affected.add(other)
    return list(affected)


def _extract_concepts_batch(items: list[tuple[str, str, str]]) -> list[ExtractedConceptWithSource]:
    provider = get_provider()
    results = []
    for fname, numbered_content, existing_list in items:
        try:
            prompt = EXTRACTION_PROMPT.format(
                sourceFile=fname,
                sourceContent=numbered_content,
                existingConcepts=existing_list or "None yet",
            )
            concepts_data = provider.extract_concepts(EXTRACTION_SYSTEM, prompt)
            if concepts_data is None:
                continue
            for c in concepts_data[:MAX_CONCEPTS]:
                slug = slugify(c.get("concept", "Unknown"))
                if not slug:
                    continue
                contradicted_raw = c.get("contradicted_by", [])
                contradicted = []
                if isinstance(contradicted_raw, list):
                    for item in contradicted_raw:
                        if isinstance(item, dict):
                            contradicted.append(Contradiction(
                                slug=item.get("slug", ""),
                                reason=item.get("reason", ""),
                            ))
                        elif isinstance(item, str):
                            contradicted.append(Contradiction(slug=item, reason=""))
                concept = ExtractedConcept(
                    concept=c.get("concept", "Unknown"),
                    summary=c.get("summary", ""),
                    is_new=c.get("is_new", True),
                    tags=c.get("tags", []),
                    confidence=c.get("confidence", 1.0),
                    provenance_state=c.get("provenance_state", "extracted"),
                    contradicted_by=contradicted,
                )
                results.append(ExtractedConceptWithSource(
                    concept=concept,
                    sourceFile=fname,
                    sourceContent=numbered_content,
                    sourceLines=numbered_content.count("\n") + 1,
                ))
        except LLMError as e:
            print(f"  [warn] Extraction failed for {fname}: {e}")
    return results


def _merge_concepts(
    extracted: list[ExtractedConceptWithSource],
    frozen_slugs: set[str],
) -> list[MergedConcept]:
    slug_map: dict[str, MergedConcept] = {}
    for item in extracted:
        slug = slugify(item.concept.concept)
        if not slug:
            continue
        if slug in frozen_slugs:
            continue
        if slug not in slug_map:
            slug_map[slug] = MergedConcept(
                slug=slug,
                title=item.concept.concept,
                summary=item.concept.summary,
                isNew=item.concept.is_new,
                tags=item.concept.tags,
                confidence=item.concept.confidence,
                provenanceState=item.concept.provenance_state,
                contradictedBy=item.concept.contradicted_by,
                sourceFiles=[],
                sourceContents=[],
                combinedContent="",
            )
        mc = slug_map[slug]
        if item.sourceFile not in mc.sourceFiles:
            mc.sourceFiles.append(item.sourceFile)
            mc.sourceContents.append((item.sourceFile, item.sourceContent))
        mc.combinedContent = f"Combined sources: {', '.join(mc.sourceFiles)}"
        mc.confidence = min(mc.confidence, item.concept.confidence)
        mc.provenanceState = "merged" if len(mc.sourceFiles) > 1 else mc.provenanceState
        if item.concept.contradicted_by:
            existing_slugs = {c.slug for c in mc.contradictedBy}
            for c in item.concept.contradicted_by:
                if c.slug not in existing_slugs:
                    mc.contradictedBy.append(c)
                    existing_slugs.add(c.slug)
    return list(slug_map.values())


def _load_page_content(root, slug):
    concepts_dir = os.path.join(root, CONCEPTS_DIR)
    path = os.path.join(concepts_dir, f"{slug}.md")
    raw = read_maybe(path)
    if not raw:
        return None, Frontmatter()
    fm, body = parse_frontmatter(raw)
    return body, fm


def _load_related_page_contents(root, all_slugs, current_slug, max_count=5):
    related = []
    for slug in sorted(all_slugs):
        if slug == current_slug:
            continue
        if len(related) >= max_count:
            break
        body, fm = _load_page_content(root, slug)
        if body:
            title = fm.title or slug
            related.append(f"--- {title} ---\n{body[:2000]}")
    return "\n\n".join(related)


def _generate_page(
    root: str,
    concept: MergedConcept,
    known_slugs: set[str],
) -> str | None:
    provider = get_provider()

    # Existing page content for update context
    existing_body, existing_fm = _load_page_content(root, concept.slug)
    existing_page_section = ""
    if existing_body:
        existing_page_section = f"-- Existing page to update --\n{existing_body}\n"

    # Related pages content
    related_content = _load_related_page_contents(root, known_slugs, concept.slug, 5)

    # Format combined source content with numbered lines + fair-share truncation
    budget = resolve_prompt_budget_chars()
    formatted_sources = format_source_sections(
        concept.sourceContents, budget
    )

    prompt = PAGE_GEN_PROMPT.format(
        title=concept.title,
        summary=concept.summary,
        sources=", ".join(concept.sourceFiles),
        relatedPages=related_content or "None",
        existingPageSection=existing_page_section,
        combinedContent=formatted_sources or "No source content available.",
    )
    try:
        content = provider.complete(PAGE_GEN_SYSTEM, prompt)
    except LLMError as e:
        print(f"  [warn] Page generation failed for {concept.slug}: {e}")
        return None

    content = resolve_wikilinks(content, concept.slug, known_slugs)

    now = now_iso()
    fm = Frontmatter(
        title=concept.title,
        summary=concept.summary,
        sources=concept.sourceFiles,
        kind="concept",
        createdAt=existing_fm.createdAt or now,
        updatedAt=now,
        confidence=concept.confidence,
        provenanceState=concept.provenanceState,
        contradictedBy=concept.contradictedBy,
        tags=concept.tags,
    )
    fm_str = format_frontmatter(fm)
    return fm_str + content + "\n"


def _get_existing_concept_slugs(concepts_dir: str) -> set[str]:
    if not os.path.isdir(concepts_dir):
        return set()
    return {f[:-3] for f in os.listdir(concepts_dir) if f.endswith(".md")}


def _persist_frozen_slugs(state, compiled_sources: set[str]) -> None:
    if not state.frozenSlugs:
        return
    concept_to_sources = build_concept_to_sources_map(state)
    remaining = []
    for slug in state.frozenSlugs:
        owners = concept_to_sources.get(slug, [])
        compiled_owners = [s for s in owners if s in compiled_sources]
        if compiled_owners != owners:
            remaining.append(slug)
    state.frozenSlugs = remaining


def _refresh_embeddings(root: str) -> None:
    concepts_dir = os.path.join(root, CONCEPTS_DIR)
    if not os.path.isdir(concepts_dir):
        return
    try:
        provider = get_provider()
        store = EmbeddingStore(root)
        store.load()
        for fname in os.listdir(concepts_dir):
            if not fname.endswith(".md"):
                continue
            slug = fname[:-3]
            raw = read_maybe(os.path.join(concepts_dir, fname))
            if not raw:
                continue
            fm, body = parse_frontmatter(raw)
            if fm.orphaned:
                store.remove_page(slug)
                continue
            page_text = body[:8000]
            try:
                page_vec = provider.embed(page_text, input_type="document")
            except (NotImplementedError, Exception):
                return
            chunks = split_into_chunks(body)
            chunk_vecs = []
            for ct in chunks:
                try:
                    cv = provider.embed(ct, input_type="document")
                    chunk_vecs.append((ct, cv))
                except Exception:
                    continue
            store.update_page(
                slug=slug,
                title=fm.title or slug,
                summary=fm.summary or body[:200],
                body=body,
                page_vec=page_vec,
                chunk_vecs=chunk_vecs,
            )
        store.model = getattr(provider, "embed_model", "default")
        if store.entries:
            store.dimensions = len(store.entries[0]["vector"])
        store.save()
    except Exception as e:
        pass


def compile_wiki(root: str, concurrency: int | None = None) -> CompileResult:
    result = CompileResult()
    t0 = time.monotonic()
    concurrency = concurrency or DEFAULT_COMPILE_CONCURRENCY

    ensure_dirs(root)
    if not acquire_lock(root):
        raise RuntimeError("Could not acquire lock")

    try:
        state = load_state(root)
        sources = _load_sources(root)
        result.sourceCount = len(sources)
        concepts_dir = os.path.join(root, CONCEPTS_DIR)

        changes, source_map, _ = _detect_changes(state, sources)
        existing_slugs = _get_existing_concept_slugs(concepts_dir)
        concept_to_sources = build_concept_to_sources_map(state)

        # Handle deletions + frozen slugs
        frozen_slugs = set(state.frozenSlugs)
        deleted_sources = [c.file for c in changes if c.type == ChangeType.DELETED]
        for fname in deleted_sources:
            if fname not in state.sources:
                continue
            for slug in state.sources[fname].concepts:
                other_owners = [
                    s for s in concept_to_sources.get(slug, [])
                    if s != fname and s not in deleted_sources
                ]
                if not other_owners:
                    orphan_page(root, slug, concepts_dir)
                else:
                    frozen_slugs.add(slug)
            del state.sources[fname]

        # Affected sources detection
        direct_changes = [c for c in changes if c.type in (ChangeType.NEW, ChangeType.CHANGED)]
        affected = _find_affected_sources(state, direct_changes, concept_to_sources)
        for fname in affected:
            if fname not in source_map:
                continue
            if not any(c.file == fname for c in changes):
                changes.append(SourceChange(file=fname, type=ChangeType.CHANGED))

        to_compile = [
            c.file for c in changes
            if c.type in (ChangeType.NEW, ChangeType.CHANGED)
            and c.file in source_map
        ]

        if not to_compile and not deleted_sources:
            # No-op: still refresh index
            index_content = generate_index(
                root, result.sourceCount,
                len([f for f in os.listdir(concepts_dir) if f.endswith(".md")]) if os.path.isdir(concepts_dir) else 0,
                int((time.monotonic() - t0) * 1000),
            )
            atomic_write(os.path.join(root, INDEX_FILE), index_content)
            result.durationMs = int((time.monotonic() - t0) * 1000)
            return result

        # Extraction phase
        all_extracted: list[ExtractedConceptWithSource] = []
        to_extract = []
        for fname in to_compile:
            content = source_map[fname]
            numbered = numbered_lines(content)
            to_extract.append((fname, numbered, list(existing_slugs)))

        compiled_sources: set[str] = set()

        if to_extract:
            print(f"Extracting concepts from {len(to_extract)} sources...")
            for i in range(0, len(to_extract), concurrency):
                batch = to_extract[i:i + concurrency]
                extracted = _extract_concepts_batch(batch)
                all_extracted.extend(extracted)

                # Per-source state persistence (crash safety)
                for fname, _, _ in batch:
                    compiled_sources.add(fname)
                    content = source_map[fname]
                    h = sha256(content)
                    extracted_slugs = list(set(
                        slugify(e.concept.concept) for e in extracted
                        if e.sourceFile == fname and slugify(e.concept.concept)
                    ))
                    if not extracted_slugs:
                        h = ""  # Empty hash trick, retry next compile
                        # Preserve old concept list for dependency tracking
                        if fname in state.sources:
                            old_concepts = state.sources[fname].concepts
                            extracted_slugs = old_concepts
                            # Freeze old slugs to preserve pages during retry cycle
                            for old_slug in old_concepts:
                                frozen_slugs.add(old_slug)
                    state.sources[fname] = SourceState(
                        hash=h,
                        concepts=extracted_slugs,
                        compiledAt=now_iso(),
                    )

            print(f"  Extracted {len(all_extracted)} concepts total")

            # Merge with frozen slug skip
            merged = _merge_concepts(all_extracted, frozen_slugs)
            print(f"  Merged into {len(merged)} unique concepts")

            # Late affected sources: freshly discovered overlaps
            new_concept_slugs = set()
            new_slugs_by_source: dict[str, set[str]] = {}
            for e in all_extracted:
                s = slugify(e.concept.concept)
                new_concept_slugs.add(s)
                new_slugs_by_source.setdefault(e.sourceFile, set()).add(s)

            late_affected = set()
            for fname, slugs in new_slugs_by_source.items():
                for slug in slugs:
                    all_owners = concept_to_sources.get(slug, [])
                    for owner in all_owners:
                        if owner != fname and owner not in compiled_sources:
                            late_affected.add(owner)
            for fname in late_affected:
                if fname in source_map and fname not in compiled_sources:
                    print(f"  [late-affected] {fname} shares a concept, re-extracting")
                    numbered = numbered_lines(source_map[fname])
                    late_batch = [(fname, numbered, list(new_concept_slugs | existing_slugs))]
                    late_extracted = _extract_concepts_batch(late_batch)
                    all_extracted.extend(late_extracted)
                    compiled_sources.add(fname)
                    h = sha256(source_map[fname])
                    extracted_slugs = list(set(
                        slugify(e.concept.concept) for e in late_extracted
                        if e.sourceFile == fname and slugify(e.concept.concept)
                    ))
                    state.sources[fname] = SourceState(
                        hash=h if extracted_slugs else "",
                        concepts=extracted_slugs or state.sources.get(fname, SourceState("", [], "")).concepts,
                        compiledAt=now_iso(),
                    )
                    # Re-merge with late extractions
                    merged = _merge_concepts(all_extracted, frozen_slugs)

            # Page generation phase
            all_slugs = existing_slugs | {m.slug for m in merged}

            for concept in merged:
                concept_path = os.path.join(concepts_dir, f"{concept.slug}.md")
                page = _generate_page(root, concept, all_slugs)
                if page is None:
                    continue
                # Validate: must have title (frontmatter) and body
                if "---" not in page or len(page.strip().split("---", 2)) < 3:
                    print(f"  [warn] {concept.slug}: page validation failed (no frontmatter), skipping")
                    continue
                _, body_section = page.strip().split("---", 2)[2].strip(), ""
                if not page.strip().split("---", 2)[2].strip():
                    print(f"  [warn] {concept.slug}: page validation failed (empty body), skipping")
                    continue
                atomic_write(concept_path, page)
                result.changed.append(concept.slug)
                if concept.slug in existing_slugs:
                    result.updated.append(concept.slug)
                else:
                    result.created.append(concept.slug)

            result.pageCount = len(
                [f for f in os.listdir(concepts_dir) if f.endswith(".md")]
            ) if os.path.isdir(concepts_dir) else 0

        # Persist frozen slugs (unfreeze if all owners compiled)
        _persist_frozen_slugs(state, compiled_sources)

        # Resolve interlinks (two-pass)
        if result.changed:
            title_index = build_title_index(root, concepts_dir)
            all_current_slugs = set()
            if os.path.isdir(concepts_dir):
                all_current_slugs = {f[:-3] for f in os.listdir(concepts_dir) if f.endswith(".md")}
            resolve_links(root, concepts_dir, result.changed, result.created)

        # Update orphan status only for deleted-source orphans (not interlink-based)
        # (interlink-based orphan marking is not part of minimal design)

        # Generate index (orphan pages excluded by generate_index)
        index_content = generate_index(
            root, result.sourceCount, result.pageCount,
            int((time.monotonic() - t0) * 1000),
        )
        atomic_write(os.path.join(root, INDEX_FILE), index_content)

        result.durationMs = int((time.monotonic() - t0) * 1000)

        # Refresh embeddings
        if result.changed:
            _refresh_embeddings(root)

        # Log with created vs updated distinction
        created_str = ", ".join(f"[[{s}]]" for s in result.created)
        updated_str = ", ".join(f"[[{s}]]" for s in result.updated)
        detail_parts = []
        if created_str:
            detail_parts.append(f"Created: {created_str}")
        if updated_str:
            detail_parts.append(f"Updated: {updated_str}")
        detail = f"{len(result.changed)} pages changed"
        if detail_parts:
            detail += " (" + "; ".join(detail_parts) + ")"

        append_log(root, {
            "action": "compile",
            "sourceCount": result.sourceCount,
            "pageCount": result.pageCount,
            "durationMs": result.durationMs,
            "created": result.created,
            "updated": result.updated,
            "sources": [fname for fname, _ in sources],
            "deletedSources": deleted_sources,
        })

        result.sourceCount = len(sources)
        save_state(root, state)

        return result
    finally:
        release_lock(root)
