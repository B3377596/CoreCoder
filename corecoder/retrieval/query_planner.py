"""Retrieval query construction from Retrieval V2 planning artifacts."""

from __future__ import annotations

from corecoder.retrieval.models import RetrievalPlan, RetrievalQuery, TaskUnderstanding
from corecoder.retrieval.task_understanding import TaskUnderstandingAnalyzer


class RetrievalQueryPlanner:
    """Build a RetrievalQuery from TaskUnderstanding and RetrievalPlan."""

    def __init__(self, analyzer: TaskUnderstandingAnalyzer | None = None):
        self._analyzer = analyzer or TaskUnderstandingAnalyzer()

    def plan(
        self,
        understanding: TaskUnderstanding,
        plan: RetrievalPlan,
    ) -> RetrievalQuery:
        symbols, concepts, likely_files = self._analyzer.build_retrieval_hints(understanding)
        return self.from_plan(plan, symbols=symbols, concepts=concepts, likely_files=likely_files)

    @staticmethod
    def from_plan(
        plan: RetrievalPlan,
        symbols: list[str] | None = None,
        concepts: list[str] | None = None,
        likely_files: list[str] | None = None,
    ) -> RetrievalQuery:
        query_symbols = list(dict.fromkeys((symbols or []) + plan.primary_symbols))
        query_concepts = list(dict.fromkeys((concepts or []) + plan.retrieval_scopes))
        query_files = list(dict.fromkeys((likely_files or []) + plan.target_files))

        strategy_extra_files = {
            "task_execution": ["__main__.py"],
            "failure_recovery": ["tests"],
        }
        task_type_extra_files = {
            "dependency_change": ["pyproject.toml", "setup.py", "requirements.txt", "setup.cfg", "__init__.py"],
            "cli_change": ["main.py", "cli.py", "app.py", "run.py", "__main__.py"],
        }
        task_type_extra_concepts = {
            "dependency_change": ["import", "dependency", "package", "install"],
            "cli_change": ["argparse", "click", "typer", "command", "argument", "flag", "option", "terminal", "console"],
        }
        for concept in task_type_extra_concepts.get(plan.task_type, []):
            if concept not in query_concepts:
                query_concepts.append(concept)
        for filename in task_type_extra_files.get(plan.task_type, []):
            if filename not in query_files:
                query_files.append(filename)
        for filename in strategy_extra_files.get(plan.retrieval_strategy, []):
            if filename not in query_files:
                query_files.append(filename)

        return RetrievalQuery(
            symbols=query_symbols[:12],
            concepts=query_concepts[:12],
            likely_files=query_files[:12],
            task_type=plan.task_type,
            expand_dependencies=plan.expansion_depth > 0,
            dependency_radius=plan.expansion_depth,
            plan=plan,
        )
