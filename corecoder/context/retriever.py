"""Symbolic repository retrieval ?*the "repository cognition" layer.

Architecture:
    Task text
      ?*    TaskIntentAnalyzer     ?*classify task type, extract symbols/concepts
      ?*    RetrievalQueryPlanner  ?*expand query based on task type
      ?*    SymbolOwnershipGraph   ?*route symbols to files, fuzzy match
      ?*    BidirectionalDepGraph  ?*expand by dependency neighborhood
      ?*    FileSummaryManager     ?*semantic summary matching
      ?*    StructuredRanker       ?*multi-factor scoring with reasoning
      ?*    ContextFragments       ?*metadata-only output (no file contents)

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

from corecoder.context.models import (
    ContextFragment,
    ContextSource,
    ContextType,
    ContextRequest,
)

# New retrieval subpackage
from corecoder.retrieval.models import (
    RetrievalContext,
    RetrievalMetrics,
    RankedFile,
    RetrievalMeta,
)
from corecoder.retrieval.repository_graph import build_repository_graph
from corecoder.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.retrieval.summaries import FileSummaryManager
from corecoder.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.retrieval.retrieval_planner import RetrievalPlanner
from corecoder.retrieval.dependency_graph import build_dependency_graph
from corecoder.retrieval.ranker import StructuredRanker
from corecoder.retrieval.models import ProjectCognition
from corecoder.codebase.indexing.index import should_skip_path, RepoIndex


# ===========================================================================
# RetrievalOptions ?*kept backward-compatible
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
# RepositoryContextRetriever ?*refactored
# ===========================================================================

class RepositoryContextRetriever:
    """Symbolic repository context retrieval.

    Uses the structured repo index (symbols.json, dependencies.json)
    plus heuristic summaries and task-aware ranking to find relevant
    files by symbolic proximity rather than raw text matching.

    Pipeline:
        1. TaskIntent analysis
        2. RetrievalQuery planning
        3. Symbol routing (symbol ?*file)
        4. Dependency expansion (follow imports)
        5. Semantic summary ranking
        6. Metadata-only ContextFragment output

    Usage:
        retriever = RepositoryContextRetriever(working_dir="/path/to/repo")
        fragments = retriever.retrieve(request)
    """

    def __init__(self, working_dir: str = ".", repo_index: RepoIndex | None = None):
        self._working_dir = Path(working_dir)
        self._index_dir = self._working_dir / ".corecoder"

        # Optional RepoIndex ?*when provided, data is read from it instead
        # of re-reading .corecoder/*.json from disk.
        self._repo_index = repo_index

        # Core indexes (lazy-loaded, may come from RepoIndex)
        self._symbols_json: dict[str, Any] = {}
        self._dependencies_json: dict[str, Any] = {}
        self._summary: str = ""
        self._loaded = False
        self._known_files_normalized: list[str] = []
        self._known_file_set: set[str] = set()
        self._basename_to_paths: dict[str, list[str]] = {}

        # New retrieval components
        self._symbol_graph = SymbolOwnershipGraph()
        self._summary_manager = FileSummaryManager(str(working_dir))
        self._dep_graph = None  # Lazy: built after loading
        self._intent_analyzer = TaskIntentAnalyzer()
        self._retrieval_planner = RetrievalPlanner()
        self._query_planner = RetrievalQueryPlanner()
        self._ranker: StructuredRanker | None = None  # Lazy: built after loading
        self._repository_graph = None

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
        """Retrieve repository context ?*metadata only, no file contents.

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

        # ---- Stage 1: Task Understanding ----
        understanding = self._intent_analyzer.understand(
            task_title=request.task_title,
            task_description=request.task_description,
            goal=request.goal,
        )
        intent = self._intent_analyzer.analyze(
            task_title=request.task_title,
            task_description=request.task_description,
            goal=request.goal,
        )
        retrieval_context = self._build_retrieval_context(request)

        # ---- Stage 2: Retrieval Planning ----
        plan = self._retrieval_planner.plan(understanding, retrieval_context)
        retrieval_context.current_plan = plan
        query = self._query_planner.from_plan(plan, intent)

        # ---- Mode Switch: understanding vs execution ----
        if intent.family == "understanding":
            return self._retrieve_understanding(request, intent, query, opts, t0)

        # ---- Stage 3: Graph-aware candidate collection ----
        candidates = self._collect_candidates(query, retrieval_context)

        # ---- Stage 4: Structured ranking ----
        ranked = self._ranker.rank(candidates, query, intent, retrieval_context)

        # ---- Stage 5: Adaptive retrieval ----
        if (not ranked or ranked[0].score < 0.2) and not retrieval_context.requested_more_context:
            missing_symbols = [
                symbol for symbol in plan.primary_symbols
                if not self._symbol_graph.lookup(symbol) and not self._symbol_graph.fuzzy_search(symbol, limit=1)
            ]
            retrieval_context.request_more_context(
                reason="low_confidence_initial_retrieval",
                additional_scopes=plan.retrieval_scopes[:3],
                missing_symbols=missing_symbols,
                requested_files=plan.target_files[:3],
            )
            plan = self._retrieval_planner.plan(understanding, retrieval_context)
            query = self._query_planner.from_plan(plan, intent)
            candidates = self._collect_candidates(query, retrieval_context)
            ranked = self._ranker.rank(candidates, query, intent, retrieval_context)

        # ---- Stage 6: Graph expansion ----
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
                    lines.append(f"- {rf.filepath} ?*{summary.purpose}")
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
                dep_entries: list[str] = []
                shown = 0
                for rf in top_files:
                    filepath = rf.filepath
                    for f, imps in internal.items():
                        if f.replace("\\", "/") == filepath and imps:
                            dep_entries.append(f"- {filepath} imports: {', '.join(imps[:5])}")
                            shown += 1
                            break
                    if shown >= 10:
                        break
                if dep_entries:
                    dep_entries.insert(0, "## Key Imports")
                    fragments.append(ContextFragment(
                        source=ContextSource.DEPENDENCY_GRAPH,
                        type=ContextType.DEPENDENCY,
                        content="\n".join(dep_entries),
                        priority=7,
                        relevance_score=0.85,
                        token_count=len("\n".join(dep_entries)) // 3,
                    ))

        # ---- Stage 8: Retrieval metadata (for debugging) ----
        retrieval_time_ms = (time.time() - t0) * 1000.0
        self._last_retrieval_meta = RetrievalMeta(
            query=query,
            intent=intent,
            understanding=understanding,
            plan=plan,
            retrieval_context=retrieval_context,
            total_files_considered=len(candidates),
            total_files_ranked=len(ranked),
            retrieval_time_ms=retrieval_time_ms,
            pipeline_stages=[
                "task_understanding",
                "retrieval_planning",
                "graph_candidate_collection",
                "structured_ranking",
                "adaptive_retrieval",
                "graph_expansion",
                "fragment_assembly",
            ],
            notes=[reason.reason for reason in retrieval_context.followup_requests],
        )

        return fragments

    def get_last_retrieval_meta(self) -> RetrievalMeta | None:
        """Return metadata about the most recent retrieval (for debugging)."""
        return self._last_retrieval_meta

    def evaluate_ranking(
        self,
        expected_files: set[str],
        ranked_files: list[str],
        token_cost: int = 0,
    ) -> RetrievalMetrics:
        """Build RetrievalMetrics for a ranked file list."""
        return RetrievalMetrics.from_rankings(
            expected=expected_files,
            retrieved=ranked_files,
            token_cost=token_cost,
        )

    def _build_retrieval_context(self, request: ContextRequest) -> RetrievalContext:
        """Build state-aware retrieval inputs from a ContextRequest."""
        if isinstance(request.retrieval_context, RetrievalContext):
            return request.retrieval_context

        metadata = request.metadata or {}
        working_memory = []
        for artifact in request.completed_artifact_map.values():
            desc = artifact.get("description")
            if desc:
                working_memory.append(str(desc))
        working_memory.extend(str(x) for x in metadata.get("working_memory", []))

        plan = metadata.get("retrieval_plan")
        if not isinstance(plan, type(None)) and not hasattr(plan, "primary_symbols"):
            plan = None

        return RetrievalContext(
            user_query=request.goal or request.task_description or request.task_title,
            active_files=list(dict.fromkeys(request.focus_files + metadata.get("active_files", []))),
            active_symbols=list(dict.fromkeys(request.focus_symbols + metadata.get("active_symbols", []))),
            current_plan=plan,
            working_memory=working_memory[:12],
            previous_failures=list(dict.fromkeys(request.recent_errors + metadata.get("previous_failures", []))),
            previous_queries=list(metadata.get("previous_queries", [])),
            metadata=metadata,
        )

    def _collect_candidates(
        self,
        query,
        retrieval_context: RetrievalContext,
    ) -> list[str]:
        """Collect repository candidates using RetrievalPlan + RepositoryGraph."""
        candidates: list[str] = []
        seen: set[str] = set()
        plan = query.plan

        def add(path: str) -> None:
            normalized = path.replace("\\", "/")
            if (
                normalized
                and normalized in self._known_file_set
                and normalized not in seen
                and not should_skip_path(normalized)
            ):
                seen.add(normalized)
                candidates.append(normalized)

        for active in retrieval_context.active_files:
            add(active)

        for symbol in query.symbols:
            for si in self._symbol_graph.fuzzy_search(symbol, limit=6):
                add(si.defined_in)
            if self._repository_graph is not None:
                for path in self._repository_graph.related_files(symbol, depth=query.dependency_radius):
                    add(path)

        for filepath in query.likely_files:
            for match in self._match_likely_file(filepath):
                add(match)

        scopes = list(query.concepts)
        if plan is not None:
            scopes.extend(plan.retrieval_scopes)
        for scope in scopes:
            scope_lower = scope.lower()
            for filepath in self._known_files_normalized:
                stem = filepath.split("/")[-1].lower()
                if scope_lower in stem:
                    add(filepath)
                    continue
                summary = self._summary_manager.get(filepath)
                if summary is None:
                    continue
                search_text = (
                    summary.purpose + " " + " ".join(summary.responsibilities) + " " + summary.category
                ).lower()
                if scope_lower in search_text:
                    add(filepath)

        if plan is not None and self._repository_graph is not None:
            for symbol in plan.primary_symbols[:6]:
                for path in self._repository_graph.related_files(symbol, depth=plan.expansion_depth):
                    add(path)
            for filepath in plan.target_files[:6]:
                for match in self._match_likely_file(filepath):
                    add(match)
                    for path in self._repository_graph.related_files(match, depth=plan.expansion_depth):
                        add(path)

        if len(candidates) < 5:
            for filepath in self._known_files_normalized:
                add(filepath)
                if len(candidates) >= 8:
                    break

        return candidates

    # ------------------------------------------------------------------
    # understanding retrieval pipeline
    # ------------------------------------------------------------------

    def _retrieve_understanding(
        self,
        request: ContextRequest,
        intent,
        query,
        opts,
        t0: float,
    ) -> list[ContextFragment]:
        """Understanding retrieval pipeline.

        DOES NOT use symbol routing.  Instead, prioritizes:
        - Entrypoints (main.py, cli.py, app.py)
        - Package inits (__init__.py at shallow depth)
        - Config files (pyproject.toml, setup.py)
        - High-centrality modules (architectural hubs)
        - README files

        The goal is project OVERVIEW, not symbol localization.
        """
        fragments: list[ContextFragment] = []

        # ---- Collect understanding candidates ----
        candidates: list[str] = []

        # 1. Query-specified likely files (from planner)
        for f in query.likely_files:
            # Search the index for files matching the name
                for normalized in self._match_likely_file(f):
                    if normalized not in candidates:
                        candidates.append(normalized)

        # 2. Top-level and shallow-depth files (best for overview)
        filepaths = (
            [node.name for node in self._repository_graph.file_nodes()]
            if self._repository_graph is not None
            else list(self._symbols_json.keys())
        )
        for filepath in filepaths:
            fp = filepath.replace("\\", "/")
            if should_skip_path(fp):
                continue
            if fp in candidates:
                continue
            depth = fp.count("/")
            fname = fp.split("/")[-1].lower()
            # Shallow files with revealing names
            if depth <= 2 and (
                fname.endswith(".py")
                or fname in ("pyproject.toml", "setup.py", "setup.cfg", "makefile")
                or fname.startswith("readme")
            ):
                candidates.append(fp)

        # 3. Add architectural hubs if we have centrality data
        centrality = self._ranker.get_centrality() if self._ranker else {}
        if centrality:
            hubs = sorted(
                centrality.items(),
                key=lambda x: x[1].centrality,
                reverse=True,
            )
            for fp, _ in hubs[:10]:
                if fp not in candidates:
                    candidates.append(fp)

        # 4. Fallback: add all non-skipped files up to limit
        if len(candidates) < 3:
            for filepath in filepaths:
                fp = filepath.replace("\\", "/")
                if fp not in candidates and not should_skip_path(filepath):
                    candidates.append(fp)
                if len(candidates) >= 8:
                    break

        # ---- Understanding-mode ranking ----
        ranked = self._ranker.rank_understanding(candidates, query, intent, None)

        # ---- Build fragments ----
        top_files = ranked[:opts.max_files]

        # Project overview fragment
        if top_files:
            lines = ["## Project Overview"]
            # Generate a project cognition summary
            cognition = self._build_project_cognition(top_files)
            if cognition.architecture_summary:
                lines.append(f"\n{cognition.architecture_summary}")
            if cognition.entrypoints:
                lines.append(f"\nEntrypoints: {', '.join(cognition.entrypoints[:5])}")
            if cognition.major_components:
                lines.append(f"\nMajor components: {', '.join(cognition.major_components[:8])}")

            lines.append("\n## Key Files (use read_file to see contents)")
            for rf in top_files:
                summary = self._summary_manager.get(rf.filepath)
                symbols = rf.symbols[:5]
                if summary and summary.purpose:
                    lines.append(f"- {rf.filepath} ?*{summary.purpose}")
                    if symbols:
                        lines.append(f"  symbols: {', '.join(symbols)}")
                elif symbols:
                    lines.append(f"- {rf.filepath} ({', '.join(symbols)})")
                else:
                    lines.append(f"- {rf.filepath}")

            fragments.append(ContextFragment(
                source=ContextSource.REPOSITORY,
                type=ContextType.METADATA,
                content="\n".join(lines),
                priority=10,
                relevance_score=0.95,
                token_count=len("\n".join(lines)) // 3,
                metadata={
                    "retrieval_mode": "understanding",
                    "ranked_files": [
                        {"path": rf.filepath, "score": rf.score, "reasons": rf.reasons}
                        for rf in top_files
                    ],
                },
            ))

        # Dependency relationships (lightweight ?*only for top files)
        if self._dependencies_json:
            internal = self._dependencies_json.get("internal_imports", {})
            if internal:
                dep_entries: list[str] = []
                shown = 0
                for rf in top_files[:5]:
                    for f, imps in internal.items():
                        if f.replace("\\", "/") == rf.filepath and imps:
                            dep_entries.append(f"- {rf.filepath} imports: {', '.join(imps[:5])}")
                            shown += 1
                            break
                    if shown >= 8:
                        break
                if dep_entries:
                    dep_entries.insert(0, "## Key Dependencies")
                    fragments.append(ContextFragment(
                        source=ContextSource.DEPENDENCY_GRAPH,
                        type=ContextType.DEPENDENCY,
                        content="\n".join(dep_entries),
                        priority=6,
                        relevance_score=0.75,
                        token_count=len("\n".join(dep_entries)) // 3,
                    ))

        # Retrieval metadata
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
                "understanding_candidate_collection",
                "understanding_ranking",
                "fragment_assembly",
            ],
        )
        return fragments

    def retrieve_project_overview(self) -> ProjectCognition:
        """Dedicated project overview ?*returns a ProjectCognition struct.

        Use this for onboarding, project explanation, and architecture
        understanding queries.  Does NOT go through the retrieval pipeline.
        """
        self._ensure_loaded()
        return self._build_project_cognition(
            list(self._symbols_json.keys())
        )

    def _build_project_cognition(self, top_files: list) -> ProjectCognition:
        """Build a ProjectCognition from ranked files or file path strings."""
        from corecoder.retrieval.models import ProjectCognition

        entrypoints: list[str] = []
        components: list[str] = []
        capabilities: list[str] = []
        frameworks: list[str] = []

        for item in top_files[:15]:
            # Accept both RankedFile objects and plain strings
            fp = item.filepath if hasattr(item, 'filepath') else str(item)
            fp = fp.replace("\\", "/")
            fname = fp.split("/")[-1]
            summary = self._summary_manager.get(fp)

            # Detect entrypoints
            if fname in ("main.py", "cli.py", "app.py", "run.py", "__main__.py", "server.py"):
                entrypoints.append(fp)
            elif summary and summary.category == "cli":
                entrypoints.append(fp)

            # Detect major components (shallow-depth core modules)
            depth = fp.count("/")
            if depth <= 2 and fp.endswith(".py") and fname not in ("__init__.py",):
                if summary and summary.category in ("core_logic", "cli"):
                    components.append(fp)
                elif len(self._symbol_graph.file_symbols(fp)) >= 3:
                    components.append(fp)

            # Detect capabilities from summaries
            if summary and summary.purpose:
                capabilities.append(summary.purpose)

        # Detect frameworks from declared dependencies
        deps = self._dependencies_json.get("declared", [])
        framework_set = {"fastapi", "flask", "django", "click", "typer", "rich",
                        "sqlalchemy", "pytest", "react", "next", "express"}
        for d in deps:
            if d.lower() in framework_set:
                frameworks.append(d)

        # Architecture summary
        if entrypoints:
            arch_parts = [f"Project with {len(top_files)} files"]
            if entrypoints:
                arch_parts.append(f"entry via {entrypoints[0].split('/')[-1]}")
            if frameworks:
                arch_parts.append(f"using {', '.join(frameworks[:3])}")
            arch_summary = "; ".join(arch_parts)
        else:
            arch_summary = f"Repository with {len(top_files)} indexed files"

        return ProjectCognition(
            entrypoints=entrypoints[:5],
            major_components=components[:8],
            architecture_summary=arch_summary,
            execution_flow=entrypoints[:3],
            primary_capabilities=capabilities[:6],
            framework_hints=frameworks[:5],
        )

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
                if neighbor not in self._known_file_set:
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

    def _rebuild_file_indexes(self) -> None:
        """Build normalized path indexes for fast likely-file lookup."""
        indexed_files = {p.replace("\\", "/") for p in self._symbols_json.keys()}
        for path in self._working_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(self._working_dir)).replace("\\", "/")
            if should_skip_path(rel):
                continue
            indexed_files.add(rel)
        self._known_files_normalized = sorted(indexed_files)
        self._known_file_set = set(self._known_files_normalized)
        self._basename_to_paths = {}
        for p in self._known_files_normalized:
            base = p.split("/")[-1].lower()
            self._basename_to_paths.setdefault(base, []).append(p)

    def _match_likely_file(self, hint: str) -> list[str]:
        """Resolve likely file hints to indexed normalized repository paths."""
        if not hint:
            return []
        normalized_hint = hint.replace("\\", "/")
        hint_lower = normalized_hint.lower()
        basename = hint_lower.split("/")[-1]

        matches: list[str] = []
        direct = self._basename_to_paths.get(basename, [])
        for p in direct:
            if p not in matches:
                matches.append(p)

        if "/" in normalized_hint:
            for p in self._known_files_normalized:
                pl = p.lower()
                if pl.endswith(hint_lower):
                    if p not in matches:
                        matches.append(p)
        return matches

    # ------------------------------------------------------------------
    # index loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the repository index and build all components.

        When a RepoIndex is provided, data is read from its in-memory
        fields ?*no redundant file I/O.  Otherwise falls back to reading
        .corecoder/*.json directly from disk.
        """
        if self._loaded:
            return
        self._loaded = True

        if self._repo_index is not None:
            # Use the passed-in RepoIndex ?*avoids re-reading the same files
            self._symbols_json = dict(self._repo_index._symbols)
            self._dependencies_json = dict(self._repo_index._deps)
            self._summary = self._repo_index._summary
        else:
            # Fallback: read files directly (backward compat)
            symbols_path = self._index_dir / "symbols.json"
            if symbols_path.exists():
                try:
                    self._symbols_json = json.loads(symbols_path.read_text(encoding="utf-8"))
                except Exception:
                    self._symbols_json = {}

            deps_path = self._index_dir / "dependencies.json"
            if deps_path.exists():
                try:
                    self._dependencies_json = json.loads(deps_path.read_text(encoding="utf-8"))
                except Exception:
                    self._dependencies_json = {}

            summary_path = self._index_dir / "repository_summary.md"
            if summary_path.exists():
                try:
                    self._summary = summary_path.read_text(encoding="utf-8")
                except Exception:
                    self._summary = ""

        # ---- Build new retrieval components ----
        self._rebuild_file_indexes()

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

        # 3.5 Unified repository graph
        if self._symbols_json:
            self._repository_graph = build_repository_graph(
                self._symbols_json,
                self._dependencies_json,
            )

        # 4. Structured ranker
        self._ranker = StructuredRanker(
            symbol_graph=self._symbol_graph,
            summaries=self._summary_manager.all_summaries(),
            dep_graph=self._dep_graph,
            repository_graph=self._repository_graph,
        )

    def invalidate_cache(self) -> None:
        """Force reload of index data on next retrieval."""
        self._loaded = False
        self._symbols_json = {}
        self._dependencies_json = {}
        self._summary = ""
        self._known_files_normalized = []
        self._known_file_set = set()
        self._basename_to_paths = {}
        self._symbol_graph = SymbolOwnershipGraph()
        self._summary_manager = FileSummaryManager(str(self._working_dir))
        self._dep_graph = None
        self._ranker = None
        self._repository_graph = None
        self._last_retrieval_meta = None
