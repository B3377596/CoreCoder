"""Structured multi-factor ranker with retrieval reasoning.

Replaces: score = keyword_count
With:     score = symbol_match + summary_match + filename_match
                  + dependency_bonus + task_intent_bonus

Every ranked file carries a score breakdown and a list of "why_selected"
reasons.  This makes retrieval decisions auditable and useful for
downstream agent planning.

Weights are tuned for coding-agent use cases: symbol matching is the
strongest signal; filename matching is the weakest.
"""

from __future__ import annotations

from corecoder.orchestration.retrieval.models import (
    RankedFile,
    RetrievalQuery,
    TaskIntent,
    FileSummary,
    ArchitecturalCentrality,
)
from corecoder.orchestration.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.orchestration.retrieval.dependency_graph import BidirectionalDepGraph


# Score component weights (execution mode)
WEIGHT_SYMBOL_MATCH = 0.35
WEIGHT_SUMMARY_MATCH = 0.25
WEIGHT_FILENAME_MATCH = 0.10
WEIGHT_DEPENDENCY_BONUS = 0.15
WEIGHT_TASK_INTENT_BONUS = 0.15

# Score component weights (understanding mode)
WEIGHT_UNDERSTANDING_CENTRALITY = 0.40
WEIGHT_UNDERSTANDING_ENTRYPOINT = 0.25
WEIGHT_UNDERSTANDING_ARCHITECTURE = 0.20
WEIGHT_UNDERSTANDING_OVERVIEW = 0.15


class StructuredRanker:
    """Multi-factor file ranker for retrieval.

    Two scoring modes:
    - EXECUTION: symbol matching + summary + filename + task intent
    - UNDERSTANDING: architectural centrality + entrypoint + architecture relevance

    Usage:
        ranker = StructuredRanker(symbol_graph, summaries, dep_graph)
        ranked = ranker.rank(candidate_files, query, intent)
    """

    def __init__(
        self,
        symbol_graph: SymbolOwnershipGraph,
        summaries: dict[str, FileSummary],
        dep_graph: BidirectionalDepGraph | None = None,
    ):
        self._symbol_graph = symbol_graph
        self._summaries = summaries
        self._dep_graph = dep_graph
        # Lazily computed centrality scores
        self._centrality: dict[str, ArchitecturalCentrality] = {}
        self._centrality_computed = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def rank(
        self,
        candidate_files: list[str],
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> list[RankedFile]:
        """Rank candidate files.  Mode-aware: understanding vs execution."""
        if intent.family == "understanding":
            return self._rank_understanding(candidate_files, query, intent)
        else:
            return self._rank_execution(candidate_files, query, intent)

    def rank_understanding(
        self,
        candidate_files: list[str],
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> list[RankedFile]:
        """Public understanding-mode ranking API."""
        return self._rank_understanding(candidate_files, query, intent)

    def get_centrality(self) -> dict[str, ArchitecturalCentrality]:
        """Return cached centrality, computing it lazily if needed."""
        self._ensure_centrality()
        return self._centrality

    # ------------------------------------------------------------------
    # execution mode ranking (existing logic)
    # ------------------------------------------------------------------

    def _rank_execution(
        self,
        candidate_files: list[str],
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> list[RankedFile]:
        scored: list[RankedFile] = []
        for filepath in candidate_files:
            rf = self._score_one_execution(filepath, query, intent)
            scored.append(rf)

        if self._dep_graph and scored:
            scored = self._apply_dependency_bonus(scored)

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # understanding mode ranking
    # ------------------------------------------------------------------

    def _rank_understanding(
        self,
        candidate_files: list[str],
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> list[RankedFile]:
        """Rank files for understanding queries.

        Uses architectural centrality, entrypoint importance, and
        architecture relevance — NOT symbol matching.
        """
        self._ensure_centrality()

        scored: list[RankedFile] = []
        for filepath in candidate_files:
            rf = self._score_one_understanding(filepath, query, intent)
            scored.append(rf)

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    def _score_one_understanding(
        self,
        filepath: str,
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> RankedFile:
        """Score a file for understanding relevance.

        Dimensions:
        - Architectural centrality: how structurally important is this file?
        - Entrypoint bonus: is this a likely entrypoint?
        - Architecture relevance: does the file reveal project structure?
        - Overview relevance: how useful is this for a project overview?
        """
        filepath = filepath.replace("\\", "/")
        summary = self._summaries.get(filepath)
        symbols = self._symbol_graph.file_symbols(filepath)
        symbol_names = [s.name for s in symbols]
        reasons: list[str] = []
        breakdown: dict[str, float] = {}

        # 1. Architectural centrality (graph-based)
        cent = self._centrality.get(filepath)
        if cent:
            centrality_score = cent.centrality
            if centrality_score > 0.5:
                reasons.append(f"architectural hub (fan_in={cent.fan_in}, fan_out={cent.fan_out})")
            elif cent.is_entrypoint:
                reasons.append("entrypoint (no internal imports of this file)")
        else:
            centrality_score = self._baseline_score(filepath, summary)
        breakdown["centrality"] = round(centrality_score, 3)

        # 2. Entrypoint importance
        entrypoint_score = self._score_entrypoint_importance(filepath, summary)
        if entrypoint_score > 0.5:
            reasons.append("likely entrypoint")
        elif entrypoint_score > 0:
            reasons.append("near entrypoint")
        breakdown["entrypoint"] = round(entrypoint_score, 3)

        # 3. Architecture relevance
        arch_score = self._score_architecture_relevance(filepath, summary, query, intent)
        if arch_score > 0.5:
            reasons.append(f"reveals architecture: {summary.purpose if summary else filepath}")
        breakdown["architecture"] = round(arch_score, 3)

        # 4. Overview utility
        overview_score = self._score_overview_utility(filepath, summary, query)
        if overview_score > 0.5:
            reasons.append("useful for project overview")
        breakdown["overview"] = round(overview_score, 3)

        total = (
            centrality_score * WEIGHT_UNDERSTANDING_CENTRALITY
            + entrypoint_score * WEIGHT_UNDERSTANDING_ENTRYPOINT
            + arch_score * WEIGHT_UNDERSTANDING_ARCHITECTURE
            + overview_score * WEIGHT_UNDERSTANDING_OVERVIEW
        )

        # Ensure minimum baseline so files aren't all at 0
        baseline = self._baseline_score(filepath, summary)
        total = max(total, baseline)

        return RankedFile(
            filepath=filepath,
            score=round(total, 4),
            reasons=reasons,
            symbol_matches=[],
            summary_match=arch_score > 0 or overview_score > 0,
            dependency_neighbor=False,
            symbols=symbol_names[:8],
            score_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # understanding scoring dimensions
    # ------------------------------------------------------------------

    def _score_entrypoint_importance(
        self, filepath: str, summary: FileSummary | None
    ) -> float:
        """Score how likely this file is an entrypoint or near one."""
        fname = filepath.split("/")[-1].lower()
        cat = summary.category if summary else ""

        # Direct entrypoints
        if fname in ("main.py", "cli.py", "app.py", "run.py", "__main__.py", "server.py"):
            return 1.0
        if cat == "cli":
            return 0.8

        # Near-entrypoint signals
        if fname == "__init__.py":
            # Top-level __init__.py is architecturally important
            depth = filepath.count("/")
            if depth <= 1:
                return 0.7
            return 0.3

        if fname in ("pyproject.toml", "setup.py", "Makefile"):
            return 0.5

        # Check if summary mentions entrypoint-related terms
        if summary:
            text = (summary.purpose + " " + " ".join(summary.responsibilities)).lower()
            if any(kw in text for kw in ("entry", "cli", "command", "main", "dispatch")):
                return 0.4

        return 0.0

    def _score_architecture_relevance(
        self,
        filepath: str,
        summary: FileSummary | None,
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> float:
        """Score how much this file reveals about project architecture."""
        fname = filepath.split("/")[-1].lower()

        # Package init files reveal module structure
        if fname == "__init__.py":
            depth = filepath.count("/")
            if depth <= 2:  # Top-level or first-level package
                return 0.8
            return 0.5

        # Core logic files
        if summary and summary.category == "core_logic":
            score = 0.5
            # Boost if it exports many symbols (likely a central module)
            symbols = self._symbol_graph.file_symbols(filepath)
            if len(symbols) >= 5:
                score = min(1.0, score + 0.3)
            return score

        # Config files reveal project setup
        if summary and summary.category == "config":
            return 0.4

        return 0.1  # Everything has some structural relevance

    def _score_overview_utility(
        self,
        filepath: str,
        summary: FileSummary | None,
        query: RetrievalQuery,
    ) -> float:
        """Score how useful this file is for a high-level project overview."""
        fname = filepath.split("/")[-1].lower()

        # Top-level files are best for overview
        depth = filepath.count("/")
        if depth == 0 and fname.endswith((".py", ".md", ".toml", ".cfg")):
            return 0.9

        # README files
        if fname.startswith("readme"):
            return 1.0

        # Entrypoints
        if fname in ("main.py", "cli.py", "app.py", "__main__.py"):
            return 0.8

        # Config files
        if fname in ("pyproject.toml", "setup.py", "setup.cfg", "Makefile"):
            return 0.7

        # Core modules at shallow depth
        if summary and summary.category in ("core_logic", "cli") and depth <= 1:
            return 0.5

        if depth <= 2:
            return 0.2

        return 0.05

    # ------------------------------------------------------------------
    # execution mode scoring
    # ------------------------------------------------------------------

    def _score_one_execution(
        self,
        filepath: str,
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> RankedFile:
        """Score a single file for execution relevance (existing logic)."""
        filepath = filepath.replace("\\", "/")
        summary = self._summaries.get(filepath)
        symbols = self._symbol_graph.file_symbols(filepath)
        symbol_names = [s.name for s in symbols]

        reasons: list[str] = []
        breakdown: dict[str, float] = {}

        symbol_score = self._score_symbols(filepath, symbol_names, query, reasons)
        breakdown["symbol_match"] = symbol_score

        summary_score = self._score_summary(summary, query, reasons)
        breakdown["summary_match"] = summary_score

        filename_score = self._score_filename(filepath, query, reasons)
        breakdown["filename_match"] = filename_score

        intent_bonus = self._score_task_intent(
            filepath, summary, symbol_names, intent, reasons
        )
        breakdown["task_intent"] = intent_bonus

        total = (
            symbol_score * WEIGHT_SYMBOL_MATCH
            + summary_score * WEIGHT_SUMMARY_MATCH
            + filename_score * WEIGHT_FILENAME_MATCH
            + intent_bonus * WEIGHT_TASK_INTENT_BONUS
        )

        baseline = self._baseline_score(filepath, summary)
        total = max(total, baseline)

        return RankedFile(
            filepath=filepath,
            score=round(total, 4),
            reasons=reasons,
            symbol_matches=[
                qs for qs in query.symbols
                if any(qs.lower() in sn.lower() for sn in symbol_names)
            ],
            summary_match=summary_score > 0,
            dependency_neighbor=False,
            symbols=symbol_names[:8],
            score_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # individual scorers
    # ------------------------------------------------------------------

    def _score_symbols(
        self,
        filepath: str,
        symbol_names: list[str],
        query: RetrievalQuery,
        reasons: list[str],
    ) -> float:
        """Score by symbol name matching.

        Exact match: 1.0, partial: 0.5, no match: 0.0.
        Multiple matches increase the score.
        """
        if not query.symbols:
            return 0.0

        matched: list[str] = []
        for qs in query.symbols:
            qs_lower = qs.lower()
            for sn in symbol_names:
                sn_lower = sn.lower()
                if qs_lower == sn_lower:
                    matched.append(f"symbol `{sn}` exact match")
                elif qs_lower in sn_lower or sn_lower in qs_lower:
                    matched.append(f"symbol `{sn}` partial match for `{qs}`")

        if not matched:
            return 0.0

        # Score: each exact = 1.0, partial = 0.5, normalized to [0, 1]
        exact_count = sum(1 for m in matched if "exact" in m)
        partial_count = len(matched) - exact_count
        raw = exact_count * 1.0 + partial_count * 0.5
        score = min(1.0, raw / max(len(query.symbols), 1))

        for m in matched[:3]:
            reasons.append(m)

        return score

    def _score_summary(
        self,
        summary: FileSummary | None,
        query: RetrievalQuery,
        reasons: list[str],
    ) -> float:
        """Score by semantic summary match.

        Checks concepts against purpose + responsibilities.
        Summary match is weighted higher than filename match.
        """
        if summary is None:
            return 0.0

        search_text = (
            summary.purpose
            + " "
            + " ".join(summary.responsibilities)
            + " "
            + summary.category
        ).lower()

        matches = 0
        for concept in query.concepts:
            if concept.lower() in search_text:
                matches += 1

        if matches == 0:
            return 0.0

        score = min(1.0, matches / max(len(query.concepts), 1))
        if matches >= 2:
            reasons.append(
                f"summary matches {matches} concepts: purpose=`{summary.purpose}`"
            )
        elif matches == 1:
            reasons.append(f"summary matches concept: `{summary.purpose}`")

        return score

    def _score_filename(
        self,
        filepath: str,
        query: RetrievalQuery,
        reasons: list[str],
    ) -> float:
        """Score by filename keyword match.

        Lowest weight — filenames are weak signals.  But still useful
        when symbols are not mentioned explicitly.
        """
        stem = filepath.split("/")[-1].lower()
        # Remove extension
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]

        matches = 0
        for concept in query.concepts:
            if concept.lower() in stem:
                matches += 1
        for symbol in query.symbols:
            if symbol.lower() in stem:
                matches += 1

        if matches == 0:
            return 0.0

        reasons.append(f"filename match: `{stem}`")
        return min(1.0, matches * 0.3)

    def _score_task_intent(
        self,
        filepath: str,
        summary: FileSummary | None,
        symbol_names: list[str],
        intent: TaskIntent,
        reasons: list[str],
    ) -> float:
        """Task-type-specific scoring bonus.

        Different task types have different file preferences:
        - cli_change: prioritize main.py, cli.py, argparse handlers
        - bug_fix: prioritize test files, error handlers
        - feature_integration: prioritize entrypoints, dispatch code
        """
        if intent.type == "unknown":
            return 0.0

        fname = filepath.split("/")[-1].lower()
        cat = summary.category if summary else ""

        bonus = 0.0

        if intent.type == "cli_change":
            if cat == "cli" or fname in ("main.py", "cli.py", "app.py", "run.py"):
                bonus = 1.0
                reasons.append("CLI entrypoint (cli_change task)")
            elif any("argparse" in s.lower() or "click" in s.lower()
                     or "typer" in s.lower() for s in symbol_names):
                bonus = 0.7
                reasons.append("argument parsing (cli_change task)")

        elif intent.type == "bug_fix":
            if cat == "test":
                bonus = 0.8
                reasons.append("test file (bug_fix task)")
            elif any("error" in s.lower() or "exception" in s.lower()
                     or "validate" in s.lower() for s in symbol_names):
                bonus = 0.5
                reasons.append("error/validation code (bug_fix task)")

        elif intent.type == "feature_integration":
            if cat == "cli" or fname in ("main.py", "cli.py", "app.py",
                                          "__init__.py"):
                bonus = 1.0
                reasons.append("integration point (feature_integration task)")
            elif cat == "package":
                bonus = 0.6
                reasons.append("package init (feature_integration task)")

        elif intent.type == "refactor":
            # Refactoring: broad context — boost all non-test, non-config files
            if cat not in ("test", "config"):
                bonus = 0.3
            if cat == "core_logic":
                bonus = 0.6

        elif intent.type == "dependency_change":
            if cat == "config" or fname in (
                "pyproject.toml", "setup.py", "requirements.txt",
                "setup.cfg", "__init__.py",
            ):
                bonus = 1.0
                reasons.append("dependency config (dependency_change task)")

        elif intent.type == "test_addition":
            if cat == "test":
                bonus = 0.9
                reasons.append("test file (test_addition task)")
            elif cat == "core_logic":
                bonus = 0.5  # code that needs tests

        return bonus

    # ------------------------------------------------------------------
    # architectural centrality
    # ------------------------------------------------------------------

    def _ensure_centrality(self) -> None:
        """Compute architectural centrality from the dependency graph.

        Called lazily on first understanding-mode ranking.  Computes
        fan-in, fan-out, and composite centrality for every known file.
        """
        if self._centrality_computed:
            return
        self._centrality_computed = True

        if not self._dep_graph:
            return

        # Collect all files from the dependency graph
        all_files = set(self._dep_graph.imports.keys()) | set(self._dep_graph.imported_by.keys())

        # Compute fan-in / fan-out per file
        fan_in: dict[str, int] = {}
        fan_out: dict[str, int] = {}
        for f in all_files:
            fan_out[f] = len(self._dep_graph.imports.get(f, []))
            fan_in[f] = len(self._dep_graph.imported_by.get(f, []))

        max_fan_in = max(fan_in.values()) if fan_in else 1
        max_fan_out = max(fan_out.values()) if fan_out else 1

        for f in all_files:
            fi = fan_in.get(f, 0)
            fo = fan_out.get(f, 0)

            # Composite centrality: normalized fan-in (importance) +
            # fan-out ratio (connectivity).  Entrypoints (fan_in==0) get
            # a special bonus because they're the "face" of the project.
            centrality = (fi / max(1, max_fan_in)) * 0.6 + (fo / max(1, max_fan_out)) * 0.2

            # Entrypoint bonus: no one imports this file internally
            if fi == 0 and fo > 0:
                centrality += 0.2
            # Hub bonus: high fan-in + high fan-out
            if fi >= 3 and fo >= 3:
                centrality = min(1.0, centrality + 0.15)

            self._centrality[f] = ArchitecturalCentrality(
                filepath=f,
                fan_in=fi,
                fan_out=fo,
                is_entrypoint=(fi == 0 and fo > 0),
                is_leaf=(fo == 0),
                centrality=round(min(1.0, centrality), 4),
            )

    @staticmethod
    def compute_centrality(dep_graph: BidirectionalDepGraph | None) -> dict[str, ArchitecturalCentrality]:
        """Standalone centrality computation for external use."""
        if not dep_graph:
            return {}

        all_files = set(dep_graph.imports.keys()) | set(dep_graph.imported_by.keys())
        fan_in: dict[str, int] = {}
        fan_out: dict[str, int] = {}
        for f in all_files:
            fan_out[f] = len(dep_graph.imports.get(f, []))
            fan_in[f] = len(dep_graph.imported_by.get(f, []))

        max_fi = max(fan_in.values()) if fan_in else 1
        max_fo = max(fan_out.values()) if fan_out else 1

        result: dict[str, ArchitecturalCentrality] = {}
        for f in all_files:
            fi = fan_in.get(f, 0)
            fo = fan_out.get(f, 0)
            centrality = (fi / max(1, max_fi)) * 0.6 + (fo / max(1, max_fo)) * 0.2
            if fi == 0 and fo > 0:
                centrality += 0.2
            if fi >= 3 and fo >= 3:
                centrality = min(1.0, centrality + 0.15)

            result[f] = ArchitecturalCentrality(
                filepath=f,
                fan_in=fi,
                fan_out=fo,
                is_entrypoint=(fi == 0 and fo > 0),
                is_leaf=(fo == 0),
                centrality=round(min(1.0, centrality), 4),
            )
        return result

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------

    def _baseline_score(
        self, filepath: str, summary: FileSummary | None
    ) -> float:
        """Small baseline score based on file prominence.

        Ensures important files (entry points, core logic) appear even
        when the query has no matching symbols or concepts.
        """
        fname = filepath.split("/")[-1].lower()
        cat = summary.category if summary else ""

        # Entry points always get a baseline
        if fname in ("main.py", "cli.py", "app.py", "run.py", "__init__.py",
                     "__main__.py", "server.py"):
            return 0.15
        if cat == "cli":
            return 0.12
        if cat == "core_logic":
            return 0.08
        if cat == "config":
            return 0.06
        if cat == "package":
            return 0.10
        # Every other file gets a minimal baseline
        return 0.02

    def _apply_dependency_bonus(
        self, scored: list[RankedFile]
    ) -> list[RankedFile]:
        """Phase 2: boost files that are dependency neighbors of high-scoring files.

        This is applied AFTER individual scoring so that dependency
        structure propagates relevance.
        """
        if not self._dep_graph:
            return scored

        # Build set of high-scoring files (threshold: top 25% or score > 0.5)
        threshold = max(0.5, sorted([r.score for r in scored], reverse=True)[
            max(0, len(scored) // 4)
        ] if scored else 0.5)
        seed_files = {r.filepath for r in scored if r.score >= threshold}

        for rf in scored:
            if rf.dependency_neighbor:
                continue
            neighbors = self._dep_graph.neighborhood(rf.filepath, radius=1)
            if neighbors & seed_files and rf.filepath not in seed_files:
                rf.score += 0.15 * WEIGHT_DEPENDENCY_BONUS
                rf.dependency_neighbor = True
                rf.score_breakdown["dependency_bonus"] = round(
                    0.15 * WEIGHT_DEPENDENCY_BONUS, 4
                )
                rf.reasons.append(
                    "dependency neighbor of high-ranked file"
                )

        # Re-sort
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored
