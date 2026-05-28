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
from enum import Enum
from typing import Any


# ===========================================================================
# Intent Family — high-level retrieval mode classification
# ===========================================================================

class IntentFamily(str, Enum):
    """Top-level classification of what the user wants.

    Determines the RETRIEVAL MODE, not just ranking.  Understanding
    queries go through a fundamentally different pipeline than execution
    queries.
    """

    EXECUTION = "execution"        # "fix the sqrt bug", "add a /login endpoint"
    UNDERSTANDING = "understanding"  # "这个项目在干什么", "explain the architecture"
    NAVIGATION = "navigation"      # "where is the auth logic?", "show me config"
    EXPLANATION = "explanation"    # "how does the dispatcher work?", "why is this slow?"
    PLANNING = "planning"          # "what do I need to build X?", initial project setup


class RetrievalMode(str, Enum):
    """Which retrieval pipeline to use.

    EXECUTION → symbol routing + dependency expansion + task ranking
    UNDERSTANDING → architecture overview + entrypoint discovery + topology
    NAVIGATION → symbol lookup + file search + dependency tracing
    EXPLANATION → deep dive on specific modules + call graph
    PLANNING → project sketch + capability inventory
    """

    EXECUTION = "execution"
    UNDERSTANDING = "understanding"
    NAVIGATION = "navigation"
    EXPLANATION = "explanation"
    PLANNING = "planning"


# ===========================================================================
# Project Cognition — what the agent knows about the repository
# ===========================================================================

@dataclass
class ProjectCognition:
    """High-level understanding of a repository — its shape, not its details.

    Built once per session and refreshed on repo change.  Powers the
    understanding retrieval pipeline: when the user asks "what does this
    project do?", the answer comes from here, not from symbol routing.
    """

    entrypoints: list[str] = field(default_factory=list)
    # e.g. ["corecoder/cli.py", "corecoder/__main__.py"]

    major_components: list[str] = field(default_factory=list)
    # Top-level modules/directories that define the architecture
    # e.g. ["corecoder/agent.py", "corecoder/orchestration/", "corecoder/tools/"]

    architecture_summary: str = ""
    # One-paragraph description of the project's architecture
    # e.g. "CLI app with ReAct agent loop, DAG orchestration, and symbolic retrieval"

    execution_flow: list[str] = field(default_factory=list)
    # Ordered list of execution entry → main modules → exit
    # e.g. ["cli.py:main()", "agent.py:Agent.chat()", "llm/client.py:LLM.chat()"]

    primary_capabilities: list[str] = field(default_factory=list)
    # What can this project DO?
    # e.g. ["AI coding assistant", "DAG task orchestration", "Repository indexing"]

    framework_hints: list[str] = field(default_factory=list)
    # Detected frameworks/libraries
    # e.g. ["FastAPI", "Click", "SQLAlchemy"]


@dataclass
class RepositoryTopology:
    """Structural understanding of the repository graph.

    Centrality is measured structurally (import graph), not by keyword.
    """

    architectural_hubs: list[str] = field(default_factory=list)
    # Files with high betweenness/degree centrality in the import graph
    # These are the "structural center" of the codebase

    leaf_modules: list[str] = field(default_factory=list)
    # Files that are imported by many but import few — utility/leaf nodes

    entrypoint_paths: list[str] = field(default_factory=list)
    # Files that are NOT imported by anyone internally (true entrypoints)

    # Centrality scores per file (computed from import graph)
    centrality_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ArchitecturalCentrality:
    """Per-file structural importance metrics."""

    filepath: str
    fan_in: int = 0       # How many files import this file
    fan_out: int = 0      # How many files this file imports
    is_entrypoint: bool = False  # fan_in == 0 (no internal imports of this file)
    is_leaf: bool = False       # fan_out == 0 (this file imports nothing internal)
    centrality: float = 0.0     # Composite score 0-1


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

    Two-layer classification:
    - family: high-level retrieval mode (IntentFamily enum or string)
    - type: fine-grained task type (bug_fix, overview, architecture, ...)

    Drives retrieval ranking: different families use fundamentally
    different pipelines.  Understanding queries skip symbol routing
    entirely and go to architecture overview.
    """

    family: str = ""  # IntentFamily: execution, understanding, navigation, ...
    type: str = ""    # bug_fix, overview, architecture, feature_addition, ...

    symbols: list[str] = field(default_factory=list)  # code symbols (execution only)
    concepts: list[str] = field(default_factory=list)  # domain concepts
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
