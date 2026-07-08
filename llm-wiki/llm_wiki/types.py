from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChangeType(str, Enum):
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


@dataclass
class SourceState:
    hash: str
    concepts: list[str]
    compiledAt: str


@dataclass
class WikiState:
    version: int = 1
    sources: dict[str, SourceState] = field(default_factory=dict)
    frozenSlugs: list[str] = field(default_factory=list)
    indexHash: str = ""


@dataclass
class SourceChange:
    file: str
    type: ChangeType


@dataclass
class Contradiction:
    slug: str
    reason: str


@dataclass
class ExtractedConcept:
    concept: str
    summary: str
    is_new: bool
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    provenance_state: str = "extracted"
    contradicted_by: list[Contradiction] = field(default_factory=list)


@dataclass
class ExtractedConceptWithSource:
    concept: ExtractedConcept
    sourceFile: str
    sourceContent: str
    sourceLines: int


@dataclass
class MergedConcept:
    slug: str
    title: str
    summary: str
    isNew: bool
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    provenanceState: str = "extracted"
    contradictedBy: list[Contradiction] = field(default_factory=list)
    sourceFiles: list[str] = field(default_factory=list)
    sourceContents: list[tuple[str, str]] = field(default_factory=list)
    combinedContent: str = ""


@dataclass
class Frontmatter:
    title: str = ""
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    kind: str = "concept"
    createdAt: str = ""
    updatedAt: str = ""
    confidence: float = 1.0
    provenanceState: str = "extracted"
    contradictedBy: list[Contradiction] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    orphaned: bool = False


@dataclass
class IndexEntry:
    slug: str
    title: str
    summary: str


@dataclass
class CompileResult:
    changed: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)
    pageCount: int = 0
    sourceCount: int = 0
    durationMs: int = 0


@dataclass
class QueryResult:
    question: str = ""
    selectedPages: list[str] = field(default_factory=list)
    answer: str = ""
    saved: bool = False


@dataclass
class LLMTool:
    name: str
    description: str
    input_schema: dict
