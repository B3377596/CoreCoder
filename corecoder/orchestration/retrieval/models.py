"""Data models for the symbolic retrieval layer.

These models are the vocabulary of the "repository cognition" system.
Every structure is designed for agent reasoning, not human documentation.

Design invariants:
- All summaries are token-efficient (<200 tokens each)
- All structures are serializable (JSON-compatible)
- No embedding vectors — purely symbolic/structural
- Every ranked result carries reasoning (why_selected)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# Symbol ownership graph types
# ===========================================================================

@dataclass
class SymbolInfo:
    """A single symbol definition in the repository.

    Minimal but sufficient for routing: name, kind, location, and a
    one-line doc brief for summary matching.
    """

    name: str
    kind: str  # function, class, method, variable, module
    defined_in: str  # file path (normalized)
    line: int = 0
    signature: str = ""  # e.g. "def sqrt(x: float) -> float"
    doc_brief: str = ""  # first sentence of docstring, max 80 chars
    exported: bool = False  # is it in __all__ or a public API?


# ===========================================================================
# File summary types
# ===========================================================================

@dataclass
class FileSummary:
    """Semantic summary of a single file.

    Generated heuristically from symbol names, file path, and imports.
    Designed to fit in ~50-100 tokens for agent reasoning.

    Does NOT use embeddings — purpose and responsibilities are derived
    from structural analysis of the code.
    """

    path: str  # normalized path
    purpose: str = ""  # one-line, max 120 chars, e.g. "CLI entry point"
    responsibilities: list[str] = field(default_factory=list)  # max 5 short phrases
    key_symbols: list[str] = field(default_factory=list)  # exported/public symbols
    category: str = ""  # cli, core_logic, utility, config, test, web, data, infra
    file_type: str = ""  # python, yaml, toml, json, markdown, etc.


# ===========================================================================
# Task intent types
# ===========================================================================

@dataclass
class TaskIntent:
    """Analysis of what the user's task is trying to achieve.

    Drives retrieval ranking: different task types prioritize different
    kinds of files.  A CLI change should surface main.py/argparse code;
    a bug fix should surface files with error-prone patterns.
    """

    type: str = ""  # bug_fix, feature_addition, feature_integration,
    # refactor, rename, dependency_change, cli_change,
    # test_addition, documentation, unknown

    symbols: list[str] = field(default_factory=list)  # mentioned symbol names
    concepts: list[str] = field(default_factory=list)  # mentioned concepts
    entrypoint_related: bool = False
    affected_files: list[str] = field(default_factory=list)  # from task hints
    confidence: float = 0.5


# ===========================================================================
# Retrieval query types
# ===========================================================================

@dataclass
class RetrievalQuery:
    """Planned retrieval query — output of query planning.

    Converts raw task text into structured retrieval parameters.
    This is the bridge between "what the user asked" and "what we
    should look for in the repository index."
    """

    symbols: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    likely_files: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    expand_dependencies: bool = True
    dependency_radius: int = 1


# ===========================================================================
# Dependency graph types
# ===========================================================================

@dataclass
class BidirectionalDepGraph:
    """Dependency graph with forward AND reverse lookups.

    Avoids O(n²) reverse-dependency scanning at query time.
    Both directions are pre-computed and stored.

    Paths are normalized (forward slashes).
    """

    imports: dict[str, list[str]] = field(default_factory=dict)  # file -> [imports]
    imported_by: dict[str, list[str]] = field(default_factory=dict)  # file -> [imported_by]

    def get_imports(self, filepath: str) -> list[str]:
        return self.imports.get(filepath, [])

    def get_imported_by(self, filepath: str) -> list[str]:
        return self.imported_by.get(filepath, [])

    def neighborhood(self, seed: str, radius: int = 1) -> set[str]:
        """Expand from a seed file by following edges in both directions."""
        result: set[str] = {seed}
        frontier: set[str] = {seed}
        for _ in range(radius):
            next_frontier: set[str] = set()
            for f in frontier:
                for neighbor in self.imports.get(f, []) + self.imported_by.get(f, []):
                    if neighbor not in result:
                        result.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return result


# ===========================================================================
# Ranking types
# ===========================================================================

@dataclass
class RankedFile:
    """A file with its retrieval score and reasoning.

    Every selected file carries an explanation of WHY it was chosen.
    This is essential for downstream agent planning — the agent needs
    to know why a file is relevant to make informed read decisions.
    """

    filepath: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    symbol_matches: list[str] = field(default_factory=list)
    summary_match: bool = False
    dependency_neighbor: bool = False
    symbols: list[str] = field(default_factory=list)  # key symbols in this file

    # Scoring breakdown for observability
    score_breakdown: dict[str, float] = field(default_factory=dict)


# ===========================================================================
# Retrieval result metadata
# ===========================================================================

@dataclass
class RetrievalMeta:
    """Metadata about the retrieval process — for debugging and agent reasoning."""

    query: RetrievalQuery = field(default_factory=RetrievalQuery)
    intent: TaskIntent = field(default_factory=TaskIntent)
    total_files_considered: int = 0
    total_files_ranked: int = 0
    retrieval_time_ms: float = 0.0
    pipeline_stages: list[str] = field(default_factory=list)
