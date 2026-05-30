"""Legacy RetrievalQuery planner built on top of Retrieval V2 planning."""

from __future__ import annotations

from corecoder.retrieval.models import RetrievalContext, RetrievalPlan, RetrievalQuery, TaskIntent
from corecoder.retrieval.retrieval_planner import RetrievalPlanner


class RetrievalQueryPlanner:
    """Compatibility adapter from TaskIntent to RetrievalQuery.

    Retrieval V2 introduces ``RetrievalPlan`` as the primary planning artifact.
    ``RetrievalQuery`` is retained so existing ranking/retriever code can migrate
    incrementally without breaking call sites.
    """

    def __init__(self, planner: RetrievalPlanner | None = None):
        self._planner = planner or RetrievalPlanner()

    def plan(
        self,
        intent: TaskIntent,
        retrieval_context: RetrievalContext | None = None,
    ) -> RetrievalQuery:
        understanding = intent.understanding
        if understanding is None:
            from corecoder.retrieval.task_intent import TaskIntentAnalyzer

            understanding = TaskIntentAnalyzer().understand(goal=" ".join(intent.symbols + intent.concepts))

        retrieval_context = retrieval_context or RetrievalContext(
            user_query=understanding.goal or understanding.objective,
            active_files=intent.affected_files,
            active_symbols=intent.symbols,
        )
        plan = self._planner.plan(understanding, retrieval_context)
        return self.from_plan(plan, intent)

    @staticmethod
    def from_plan(plan: RetrievalPlan, intent: TaskIntent | None = None) -> RetrievalQuery:
        intent = intent or TaskIntent()
        legacy_extra_files = {
            "dependency_change": ["pyproject.toml", "setup.py", "requirements.txt", "setup.cfg", "__init__.py"],
            "cli_change": ["main.py", "cli.py", "app.py", "run.py", "__main__.py"],
        }
        legacy_extra_concepts = {
            "dependency_change": ["import", "dependency", "package", "install"],
            "cli_change": ["argparse", "click", "typer", "command", "argument", "flag", "option", "terminal", "console"],
        }
        concepts = list(dict.fromkeys((intent.concepts or []) + plan.retrieval_scopes))
        if intent.type in legacy_extra_concepts:
            concepts.extend(c for c in legacy_extra_concepts[intent.type] if c not in concepts)
        likely_files = list(dict.fromkeys((intent.affected_files or []) + plan.target_files))
        if intent.type in legacy_extra_files:
            likely_files.extend(f for f in legacy_extra_files[intent.type] if f not in likely_files)
        return RetrievalQuery(
            symbols=list(dict.fromkeys((intent.symbols or []) + plan.primary_symbols)),
            concepts=concepts[:12],
            likely_files=likely_files[:12],
            task_type=intent.type or plan.task_type,
            expand_dependencies=plan.expansion_depth > 0,
            dependency_radius=plan.expansion_depth,
            plan=plan,
        )
