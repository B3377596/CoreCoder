"""Symbolic repository retrieval — the "repository cognition" layer.

Architecture:
    Task text
      ↓
    TaskIntentAnalyzer     — classify task type, extract symbols/concepts
      ↓
    RetrievalQueryPlanner  — expand query based on task type
      ↓
    SymbolOwnershipGraph   — route symbols to files, fuzzy match
      ↓
    BidirectionalDepGraph  — expand by dependency neighborhood
      ↓
    FileSummaryManager     — semantic summary matching
      ↓
    StructuredRanker       — multi-factor scoring with reasoning
      ↓
    ContextFragments       — metadata-only output (no file contents)

Design invariants:
- Metadata-first: retrieve() returns file listings, symbols, dependencies.
  File CONTENTS belong to tool calls (read_file), not here.
- Grounding is separate: the retriever decides WHAT is relevant; the
  agent decides when to read file contents via tools.
- No embeddings: purely symbolic/structural retrieval.
- Token-efficient: all summaries are ~50-100 tokens.
- Explainable: every ranked file carries "why_selected" reasoning.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextSource,
    ContextType,
    ContextRequest,
)

# New retrieval subpackage
from corecoder.orchestration.retrieval.models import (
    RankedFile,
    RetrievalMeta,
)
from corecoder.orchestration.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.orchestration.retrieval.summaries import FileSummaryManager
from corecoder.orchestration.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.orchestration.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.orchestration.retrieval.dependency_graph import build_dependency_graph
from corecoder.orchestration.retrieval.ranker import StructuredRanker
from corecoder.repo.index import should_skip_path


# ===========================================================================
# RetrievalOptions — kept backward-compatible
# ===========================================================================

@dataclass
class RetrievalOptions:
    """Controls for the retrieval process.

    Compatible with the old API.  New fields are additive.
    """

    max_files: int = 10
    max_symbols: int = 20
    dependency_radius: int = 2
    include_callers: bool = True
    include_callees: bool = True
    prefer_recently_modified: bool = True


# ===========================================================================
# RepositoryContextRetriever — refactored
# ===========================================================================

class RepositoryContextRetriever:
    """Symbolic repository context retrieval.

    Uses the structured repo index (symbols.json, dependencies.json)
    plus heuristic summaries and task-aware ranking to find relevant
    files by symbolic proximity rather than raw text matching.

    Pipeline:
        1. TaskIntent analysis
        2. RetrievalQuery planning
        3. Symbol routing (symbol → file)
        4. Dependency expansion (follow imports)
        5. Semantic summary ranking
        6. Metadata-only ContextFragment output

    Usage:
        retriever = RepositoryContextRetriever(working_dir="/path/to/repo")
        fragments = retriever.retrieve(request)
    """

    def __init__(self, working_dir: str = "."):
        self._working_dir = Path(working_dir)
        self._index_dir = self._working_dir / ".corecoder"

        # Core indexes (lazy-loaded)
        self._symbols_json: dict[str, Any] = {}
        self._dependencies_json: dict[str, Any] = {}
        self._summary: str = ""
        self._loaded = False

        # New retrieval components
        self._symbol_graph = SymbolOwnershipGraph()
        self._summary_manager = FileSummaryManager(str(working_dir))
        self._dep_graph = None  # Lazy: built after loading
        self._intent_analyzer = TaskIntentAnalyzer()
        self._query_planner = RetrievalQueryPlanner()
        self._ranker: StructuredRanker | None = None  # Lazy: built after loading

        # Cache
        self._last_retrieval_meta: RetrievalMeta | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        request: ContextRequest,
        options: RetrievalOptions | None = None,
    ) -> list[ContextFragment]:
        """Retrieve repository context — metadata only, no file contents.

        Returns ContextFragments containing:
        - Relevant files with symbols and selection reasoning
        - Dependency relationships
        - Retrieval metadata (for agent planning)

        Does NOT return file contents.  The agent uses read_file tools
        to fetch contents when needed.
        """
        opts = options or RetrievalOptions()
        self._ensure_loaded()
        t0 = time.time()

        fragments: list[ContextFragment] = []

        if not self._symbol_graph.is_built:
            return fragments

        # ---- Stage 1: Task Intent Analysis ----
        intent = self._intent_analyzer.analyze(
            task_title=request.task_title,
            task_description=request.task_description,
            goal=request.goal,
        )

        # ---- Stage 2: Query Planning ----
        query = self._query_planner.plan(intent)

        # ---- Stage 3: Symbol Routing ----
        # Find files via symbol matching (the primary signal)
        symbol_files: set[str] = set()
        if query.symbols:
            for sym in query.symbols:
                matches = self._symbol_graph.fuzzy_search(sym, limit=5)
                for si in matches:
                    symbol_files.add(si.defined_in)

        # ---- Stage 4: Candidate Collection ----
        # Start with symbol-matched files, add likely files, then all known files
        candidates: list[str] = list(symbol_files)

        # Add query-specified likely files that exist
        for f in query.likely_files:
            if f not in candidates:
                # Check if this file exists in the symbol index
                for known in self._symbols_json:
                    if known.replace("\\", "/").endswith(f):
                        candidates.append(known)
                        break

        # If too few candidates from symbol routing, use semantic summary
        # matching via FileSummaryManager instead of raw keyword grep.
        if len(candidates) < 5 and (query.concepts or intent.concepts):
            all_concepts = set(c.lower() for c in (query.concepts + intent.concepts))
            for filepath in self._symbols_json:
                if should_skip_path(filepath):
                    continue
                fp = filepath.replace("\\", "/")
                if fp in candidates:
                    continue
                summary = self._summary_manager.get(fp)
                if summary is None:
                    continue
                search_text = (
                    summary.purpose + " " + " ".join(summary.responsibilities)
                ).lower()
                if any(c in search_text for c in all_concepts):
                    candidates.append(fp)

        # Still too few — add all non-skipped files as last resort
        if len(candidates) < 3:
            for filepath in self._symbols_json:
                fp = filepath.replace("\\", "/")
                if fp not in candidates and not should_skip_path(filepath):
                    candidates.append(fp)
                if len(candidates) >= 5:
                    break

        # ---- Stage 5: Structured Ranking ----
        ranked = self._ranker.rank(candidates, query, intent)

        # ---- Stage 6: Dependency Expansion ----
        if self._dep_graph and query.expand_dependencies:
            ranked = self._expand_by_dependencies(
                ranked, query.dependency_radius, opts.max_files
            )

        # ---- Stage 7: Build metadata fragments ----
        top_files = ranked[:opts.max_files]

        # 7a. Project files overview
        if top_files:
            lines = ["## Project Files (use read_file to see contents)"]
            for rf in top_files:
                summary = self._summary_manager.get(rf.filepath)
                symbols = rf.symbols[:5]
                if summary and summary.purpose:
                    lines.append(f"- {rf.filepath} — {summary.purpose}")
                    if symbols:
                        lines.append(f"  symbols: {', '.join(symbols)}")
                elif symbols:
                    lines.append(f"- {rf.filepath} ({', '.join(symbols)})")
                else:
                    lines.append(f"- {rf.filepath}")

            # Attach retrieval reasoning in metadata
            reasons_map: dict[str, list[str]] = {}
            for rf in top_files:
                if rf.reasons:
                    reasons_map[rf.filepath] = rf.reasons

            fragments.append(ContextFragment(
                source=ContextSource.REPOSITORY,
                type=ContextType.METADATA,
                content="\n".join(lines),
                priority=9,
                relevance_score=0.95,
                token_count=len("\n".join(lines)) // 3,
                metadata={
                    "ranked_files": [
                        {
                            "path": rf.filepath,
                            "score": rf.score,
                            "reasons": rf.reasons,
                        }
                        for rf in top_files
                    ],
                },
            ))

        # 7b. Dependency relationships
        if self._dependencies_json:
            internal = self._dependencies_json.get("internal_imports", {})
            if internal:
                dep_lines = ["## Key Imports"]
                shown = 0
                for rf in top_files:
                    filepath = rf.filepath
                    # Find matching key in internal_imports
                    for f, imps in internal.items():
                        if f.replace("\\", "/") == filepath and imps:
                            dep_lines.append(f"- {filepath} imports: {', '.join(imps[:5])}")
                            shown += 1
                            break
                    if shown >= 10:
                        break
                if dep_lines:
                    fragments.append(ContextFragment(
                        source=ContextSource.DEPENDENCY_GRAPH,
                        type=ContextType.DEPENDENCY,
                        content="\n".join(dep_lines),
                        priority=7,
                        relevance_score=0.85,
                        token_count=len("\n".join(dep_lines)) // 3,
                    ))

        # ---- Stage 8: Retrieval metadata (for debugging) ----
        retrieval_time_ms = (time.time() - t0) * 1000.0
        self._last_retrieval_meta = RetrievalMeta(
            query=query,
            intent=intent,
            total_files_considered=len(candidates),
            total_files_ranked=len(ranked),
            retrieval_time_ms=retrieval_time_ms,
            pipeline_stages=[
                "intent_analysis",
                "query_planning",
                "symbol_routing",
                "structured_ranking",
                "dependency_expansion",
                "fragment_assembly",
            ],
        )

        # Print debug info
        '''print(
            f"[Retriever] intent={intent.type} confidence={intent.confidence:.2f}, "
            f"symbols={query.symbols}, concepts={query.concepts}, "
            f"candidates={len(candidates)}, ranked={len(ranked)}, "
            f"top_files={[rf.filepath for rf in top_files[:5]]}, "
            f"time={retrieval_time_ms:.1f}ms"
        )'''

        return fragments

    def get_last_retrieval_meta(self) -> RetrievalMeta | None:
        """Return metadata about the most recent retrieval (for debugging)."""
        return self._last_retrieval_meta

    # ------------------------------------------------------------------
    # dependency expansion
    # ------------------------------------------------------------------

    def _expand_by_dependencies(
        self,
        ranked: list[RankedFile],
        radius: int,
        max_files: int,
    ) -> list[RankedFile]:
        """Expand ranked files by following dependency edges.

        Files that are dependency neighbors of top-ranked files get a
        bonus and are included if not already present.
        """
        if not self._dep_graph or radius <= 0:
            return ranked

        existing = {rf.filepath for rf in ranked}
        top_half = {rf.filepath for rf in ranked[: max(3, len(ranked) // 3)]}

        new_files: list[RankedFile] = []

        for seed in list(top_half)[:5]:
            neighbors = self._dep_graph.neighborhood(seed, radius)
            for neighbor in neighbors:
                if neighbor in existing:
                    continue
                if should_skip_path(neighbor):
                    continue
                existing.add(neighbor)
                summary = self._summary_manager.get(neighbor)
                symbols = self._symbol_graph.file_symbols(neighbor)
                new_files.append(RankedFile(
                    filepath=neighbor,
                    score=0.2,  # Base score for dependency neighbors
                    reasons=["dependency neighbor of top-ranked file"],
                    dependency_neighbor=True,
                    symbols=[s.name for s in symbols[:5]],
                    summary_match=summary is not None and summary.purpose != "",
                    score_breakdown={"dependency_bonus": 0.2},
                ))

        ranked = ranked + new_files
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:max_files]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # index loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the repository index and build all components."""
        if self._loaded:
            return
        self._loaded = True

        # Load symbols
        symbols_path = self._index_dir / "symbols.json"
        if symbols_path.exists():
            try:
                self._symbols_json = json.loads(symbols_path.read_text(encoding="utf-8"))
            except Exception:
                self._symbols_json = {}

        # Load dependencies
        deps_path = self._index_dir / "dependencies.json"
        if deps_path.exists():
            try:
                self._dependencies_json = json.loads(deps_path.read_text(encoding="utf-8"))
            except Exception:
                self._dependencies_json = {}

        # Load summary
        summary_path = self._index_dir / "repository_summary.md"
        if summary_path.exists():
            try:
                self._summary = summary_path.read_text(encoding="utf-8")
            except Exception:
                self._summary = ""

        # ---- Build new retrieval components ----

        # 1. Symbol ownership graph
        if self._symbols_json:
            self._symbol_graph.build_from_index(self._symbols_json)

        # 2. File summaries (try cache first, then build)
        self._summary_manager.load_cache()
        if not self._summary_manager.all_summaries() and self._symbols_json:
            self._summary_manager.build(self._symbols_json)
            self._summary_manager.save_cache()

        # 3. Bidirectional dependency graph
        if self._dependencies_json:
            self._dep_graph = build_dependency_graph(self._dependencies_json)

        # 4. Structured ranker
        self._ranker = StructuredRanker(
            symbol_graph=self._symbol_graph,
            summaries=self._summary_manager.all_summaries(),
            dep_graph=self._dep_graph,
        )

    def invalidate_cache(self) -> None:
        """Force reload of index data on next retrieval."""
        self._loaded = False
        self._symbols_json = {}
        self._dependencies_json = {}
        self._summary = ""
        self._symbol_graph = SymbolOwnershipGraph()
        self._summary_manager = FileSummaryManager(str(self._working_dir))
        self._dep_graph = None
        self._ranker = None
        self._last_retrieval_meta = None
