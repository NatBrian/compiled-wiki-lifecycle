import os

SOURCES_DIR = "sources"
WIKI_DIR = "wiki"
CONCEPTS_DIR = "wiki/concepts"
QUERIES_DIR = "wiki/queries"
LLMWIKI_DIR = ".llmwiki"
STATE_FILE = ".llmwiki/state.json"
LOCK_FILE = ".llmwiki/lock"
INDEX_FILE = "wiki/index.md"
LOG_FILE = "log.md"

MAX_SOURCE_CHARS = 100_000
MIN_SOURCE_CHARS = 50
DEFAULT_PROMPT_BUDGET_CHARS = 200_000
QUERY_PAGE_LIMIT = 5
DEFAULT_COMPILE_CONCURRENCY = 5
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_PROVIDER = "anthropic"
MAX_TOKENS = 4096
MIN_CONCEPTS = 3
MAX_CONCEPTS = 8
MAX_RELATED_PAGES = 5
LOG_MAX_PAGE_LINKS = 20
RETRY_COUNT = 3
RETRY_BASE_MS = 1000
RETRY_MULTIPLIER = 4

# Embedding retrieval
EMBEDDINGS_FILE = ".llmwiki/embeddings.json"
EMBEDDING_TOP_K = 15
CHUNK_TOP_K = 30
CHUNK_RERANK_KEEP = 12
CHUNK_TARGET_CHARS = 800
CHUNK_MAX_CHARS = 1400
CHUNK_MIN_CHARS = 200
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_root():
    return os.getcwd()


def resolve_path(root, relative):
    return os.path.join(root, relative)


def resolve_prompt_budget_chars() -> int:
    raw = os.environ.get("LLMWIKI_PROMPT_BUDGET_CHARS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_PROMPT_BUDGET_CHARS
