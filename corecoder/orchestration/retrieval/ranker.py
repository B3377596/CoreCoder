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
)
from corecoder.orchestration.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.orchestration.retrieval.dependency_graph import BidirectionalDepGraph


# Score component weights
WEIGHT_SYMBOL_MATCH = 0.35
WEIGHT_SUMMARY_MATCH = 0.25
WEIGHT_FILENAME_MATCH = 0.10
WEIGHT_DEPENDENCY_BONUS = 0.15
WEIGHT_TASK_INTENT_BONUS = 0.15


class StructuredRanker:
    """Multi-factor file ranker for retrieval.

    Scores each candidate file on 5 dimensions and produces a RankedFile
    with full reasoning.  Designed for symbolic/structural matching —
    no embeddings involved.

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

    def rank(
        self,
        candidate_files: list[str],
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> list[RankedFile]:
        """Rank candidate files and return sorted results with reasoning.

        Args:
            candidate_files: List of file paths to score.
            query: The planned retrieval query.
            intent: The analyzed task intent.

        Returns:
            List of RankedFile, sorted by score descending.
        """
        # Phase 1: Compute individual scores
        scored: list[RankedFile] = []
        for filepath in candidate_files:
            rf = self._score_one(filepath, query, intent)
            if rf.score > 0:
                scored.append(rf)

        # Phase 2: Apply dependency bonus across already-scored files
        if self._dep_graph and scored:
            scored = self._apply_dependency_bonus(scored)

        # Phase 3: Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)

        return scored

    def _score_one(
        self,
        filepath: str,
        query: RetrievalQuery,
        intent: TaskIntent,
    ) -> RankedFile:
        """Score a single file on all dimensions."""
        filepath = filepath.replace("\\", "/")
        summary = self._summaries.get(filepath)
        symbols = self._symbol_graph.file_symbols(filepath)
        symbol_names = [s.name for s in symbols]

        reasons: list[str] = []
        breakdown: dict[str, float] = {}

        # 1. Symbol match score
        symbol_score = self._score_symbols(filepath, symbol_names, query, reasons)
        breakdown["symbol_match"] = symbol_score

        # 2. Semantic summary score
        summary_score = self._score_summary(summary, query, reasons)
        breakdown["summary_match"] = summary_score

        # 3. Filename score
        filename_score = self._score_filename(filepath, query, reasons)
        breakdown["filename_match"] = filename_score

        # 4. Task intent bonus (preliminary — dependency bonus applied later)
        intent_bonus = self._score_task_intent(
            filepath, summary, symbol_names, intent, reasons
        )
        breakdown["task_intent"] = intent_bonus

        # Weighted total
        total = (
            symbol_score * WEIGHT_SYMBOL_MATCH
            + summary_score * WEIGHT_SUMMARY_MATCH
            + filename_score * WEIGHT_FILENAME_MATCH
            + intent_bonus * WEIGHT_TASK_INTENT_BONUS
            # Dependency bonus applied in phase 2
        )

        return RankedFile(
            filepath=filepath,
            score=round(total, 4),
            reasons=reasons,
            symbol_matches=[
                qs for qs in query.symbols
                if any(qs.lower() in sn.lower() for sn in symbol_names)
            ],
            summary_match=summary_score > 0,
            dependency_neighbor=False,  # set in phase 2
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
