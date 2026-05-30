"""Retrieval planning for Retrieval V2."""

from __future__ import annotations

from dataclasses import dataclass

from corecoder.retrieval.models import RetrievalContext, RetrievalPlan, TaskUnderstanding


@dataclass
class PlanHeuristics:
    default_depth: int = 1
    architecture_depth: int = 2
    explanation_depth: int = 3
    recovery_depth: int = 2


class RetrievalPlanner:
    """Convert user understanding + runtime state into a RetrievalPlan."""

    def __init__(self, heuristics: PlanHeuristics | None = None):
        self._heuristics = heuristics or PlanHeuristics()

    def plan(
        self,
        understanding: TaskUnderstanding,
        retrieval_context: RetrievalContext | None = None,
    ) -> RetrievalPlan:
        retrieval_context = retrieval_context or RetrievalContext(user_query=understanding.goal)
        query = understanding.goal.lower()
        scopes = list(dict.fromkeys(
            understanding.likely_modules
            + retrieval_context.active_symbols
            + [self._basename_without_ext(f) for f in retrieval_context.active_files]
        ))
        symbols = list(dict.fromkeys(
            [e.name for e in understanding.entities if e.kind in {"class", "function", "method", "symbol"}]
            + retrieval_context.active_symbols
        ))
        target_files = list(dict.fromkeys(
            [e.name.replace("\\", "/") for e in understanding.entities if e.kind == "file"]
            + retrieval_context.active_files
        ))

        strategy = "balanced"
        task_type = "general"
        depth = self._heuristics.default_depth
        reasoning: list[str] = []
        required_context: list[str] = []

        if any(term in query for term in ("overview", "architecture", "onboarding", "what does this")):
            strategy = "architecture_scan"
            task_type = "architecture_understanding"
            depth = self._heuristics.architecture_depth
            reasoning.append("broad overview query detected")
            required_context.extend(["entrypoints", "major_components", "architecture_hubs"])
        elif any(term in query for term in ("where is", "locate", "which file", "show me")):
            strategy = "targeted_lookup"
            task_type = "symbol_navigation"
            reasoning.append("navigation query detected")
            required_context.extend(["symbol_definitions", "file_locations"])
        elif any(term in query for term in ("how does", "why", "deep dive")):
            strategy = "causal_trace"
            task_type = "implementation_explanation"
            depth = self._heuristics.explanation_depth
            reasoning.append("explanation query detected")
            required_context.extend(["dependency_paths", "related_symbols"])
        elif retrieval_context.previous_failures:
            strategy = "failure_recovery"
            task_type = "failure_investigation"
            depth = self._heuristics.recovery_depth
            reasoning.append("previous failures available, widening search")
            required_context.extend(["failing_modules", "error_related_files"])
        else:
            strategy = "task_execution"
            task_type = "targeted_change"
            reasoning.append("defaulting to focused execution search")
            required_context.extend(["active_files", "primary_symbols"])

        if retrieval_context.requested_more_context:
            depth = min(4, depth + 1)
            reasoning.append("adaptive expansion requested more context")
            for followup in retrieval_context.followup_requests:
                scopes.extend(followup.additional_scopes)
                symbols.extend(followup.missing_symbols)
                target_files.extend(followup.requested_files)

        scopes = list(dict.fromkeys([scope for scope in scopes if scope]))
        symbols = list(dict.fromkeys([symbol for symbol in symbols if symbol]))
        target_files = list(dict.fromkeys([f for f in target_files if f]))

        return RetrievalPlan(
            task_type=task_type,
            objective=understanding.objective or understanding.goal,
            primary_symbols=symbols[:12],
            retrieval_scopes=scopes[:12],
            expansion_depth=depth,
            retrieval_strategy=strategy,
            target_files=target_files[:12],
            required_context=required_context,
            plan_reasoning=reasoning,
        )

    @staticmethod
    def _basename_without_ext(path: str) -> str:
        filename = path.replace("\\", "/").split("/")[-1]
        return filename.rsplit(".", 1)[0] if "." in filename else filename
