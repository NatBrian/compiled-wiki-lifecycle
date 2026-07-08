import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="LLM Wiki - compile and query")
    sub = parser.add_subparsers(dest="command")

    compile_cmd = sub.add_parser("compile", help="Compile wiki from sources")
    compile_cmd.add_argument("--concurrency", type=int, default=5, help="Concurrent extractions")
    compile_cmd.add_argument("--dir", type=str, default=".", help="Wiki root directory")

    query_cmd = sub.add_parser("query", help="Query the wiki")
    query_cmd.add_argument("question", type=str, help="Question to answer")
    query_cmd.add_argument("--save", action="store_true", help="Save query result")
    query_cmd.add_argument("--dir", type=str, default=".", help="Wiki root directory")

    test_cmd = sub.add_parser("test", help="Run self-test")

    args = parser.parse_args()

    if args.command == "compile":
        from .compiler import compile_wiki
        root = os.path.abspath(args.dir)
        result = compile_wiki(root, concurrency=args.concurrency)
        print(json.dumps({
            "changed": result.changed,
            "created": result.created,
            "updated": result.updated,
            "deleted": result.deleted,
            "frozen": result.frozen,
            "pageCount": result.pageCount,
            "sourceCount": result.sourceCount,
            "durationMs": result.durationMs,
        }, indent=2))

    elif args.command == "query":
        from .query import query
        root = os.path.abspath(args.dir)
        result = query(root, args.question, save=args.save)
        print(f"\nSelected pages: {result.selectedPages}")
        print(f"\nAnswer:\n{result.answer}")

    elif args.command == "test":
        _run_test()

    else:
        parser.print_help()


def _run_test():
    import tempfile
    import uuid
    from .compiler import compile_wiki
    from .query import query
    from .utils import read_maybe, parse_frontmatter, load_state, sha256, atomic_write
    from .llm import set_provider
    from .llm import OpenAIProvider

    root = tempfile.mkdtemp(prefix="llm-wiki-test-")

    provider = OpenAIProvider()
    set_provider(provider)

    sources_dir = os.path.join(root, "sources")
    os.makedirs(sources_dir, exist_ok=True)

    def write_source(fname, content):
        atomic_write(os.path.join(sources_dir, fname), content)

    write_source("test1.md", "# Test Source 1\n\nMachine learning is a subset of artificial intelligence.")
    write_source("test2.md", "# Test Source 2\n\nDeep learning uses neural networks with multiple layers.")

    print("Compile 1...")
    result = compile_wiki(root)
    print(f"  Changed: {len(result.changed)}, Created: {len(result.created)}, Duration: {result.durationMs}ms")

    concepts_dir = os.path.join(root, "wiki/concepts")
    pages = [f for f in os.listdir(concepts_dir) if f.endswith(".md")] if os.path.isdir(concepts_dir) else []
    print(f"  Pages: {len(pages)}")
    for p in pages:
        raw = read_maybe(os.path.join(concepts_dir, p))
        fm, body = parse_frontmatter(raw)
        print(f"    {p}: title={fm.title}, sources={fm.sources}")

    index_raw = read_maybe(os.path.join(root, "wiki/index.md"))
    assert index_raw, "Index should exist"
    print("  Index: OK")

    state = load_state(root)
    assert len(state.sources) == 2, f"Expected 2 sources, got {len(state.sources)}"
    print("  State: OK")

    result2 = compile_wiki(root)
    assert len(result2.changed) == 0, f"Expected 0 changes, got {len(result2.changed)}"
    print("  No-op compile: OK")

    write_source("test1.md", "# Test Source 1 (edited)\n\nMachine learning is a subset of AI. It involves algorithms.")
    result3 = compile_wiki(root)
    # Now verify no errors
    print(f"  Incremental compile: OK ({len(result3.changed)} pages changed, {len(result3.created)} created)")

    print("\nAll tests passed!")

    import shutil
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
