"""Query planner — converts TaskIntent into a structured RetrievalQuery.

This is the "planning" step of retrieval: before looking at files,
we decide WHAT to look for.  The planner bridges the gap between
"the user asked to integrate sqrt into CLI" and "search for files
containing sqrt, CLI entrypoints, and routing logic."

Design: lightweight symbolic reasoning.  No LLM calls.  Uses task
type heuristics to expand the search beyond direct keyword matching.
"""

from __future__ import annotations

from corecoder.orchestration.retrieval.models import (
    TaskIntent,
    RetrievalQuery,
)


class RetrievalQueryPlanner:
    """Converts TaskIntent → RetrievalQuery.

    Expands the search scope based on task type:
    - bug_fix: add test files, error handlers
    - feature_integration: add entrypoints, dispatch, wiring code
    - cli_change: add arg parsing, command routing, main entrypoint
    - refactor: expand dependency radius

    Usage:
        planner = RetrievalQueryPlanner()
        query = planner.plan(intent)
    """

    # Task-type-specific search expansions
    _TYPE_EXPANSIONS: dict[str, dict] = {
        "bug_fix": {
            "extra_concepts": ["error", "exception", "validation", "edge case"],
            "extra_files": [],
            "expand_radius": 2,  # bugs often span multiple files
            "prioritize_tests": True,
        },
        "feature_integration": {
            "extra_concepts": ["dispatch", "route", "entrypoint", "interface",
                               "integration", "wiring"],
            "extra_files": ["main.py", "cli.py", "app.py", "__init__.py"],
            "expand_radius": 2,
            "prioritize_tests": False,
        },
        "feature_addition": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 1,
            "prioritize_tests": False,
        },
        "cli_change": {
            "extra_concepts": ["argparse", "click", "typer", "command", "argument",
                               "flag", "option", "terminal", "console"],
            "extra_files": ["main.py", "cli.py", "app.py", "run.py", "__main__.py"],
            "expand_radius": 1,
            "prioritize_tests": False,
        },
        "refactor": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 3,  # refactoring needs broad context
            "prioritize_tests": True,
        },
        "rename": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 2,
            "prioritize_tests": True,
        },
        "dependency_change": {
            "extra_concepts": ["import", "dependency", "package", "install"],
            "extra_files": ["pyproject.toml", "setup.py", "requirements.txt",
                            "setup.cfg", "__init__.py"],
            "expand_radius": 1,
            "prioritize_tests": False,
        },
        "test_addition": {
            "extra_concepts": ["test", "assert", "mock", "fixture", "coverage"],
            "extra_files": [],
            "expand_radius": 1,
            "prioritize_tests": True,
        },
        "documentation": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 0,
            "prioritize_tests": False,
        },
        "unknown": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 1,
            "prioritize_tests": False,
        },
    }

    def plan(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan a retrieval query from task intent.

        Expands:
        - concepts: adds task-type-specific concepts
        - likely_files: adds known entrypoints/config files
        - dependency_radius: adjusts based on task complexity
        """
        expansion = self._TYPE_EXPANSIONS.get(
            intent.type, self._TYPE_EXPANSIONS["unknown"]
        )

        # Merge explicit concepts with type-expanded ones
        concepts = list(dict.fromkeys(
            intent.concepts + expansion.get("extra_concepts", [])
        ))

        # Merge explicit files with type-expanded ones
        likely_files = list(dict.fromkeys(
            intent.affected_files + expansion.get("extra_files", [])
        ))

        return RetrievalQuery(
            symbols=intent.symbols,
            concepts=concepts,
            likely_files=likely_files,
            task_type=intent.type,
            expand_dependencies=expansion.get("expand_radius", 0) > 0,
            dependency_radius=expansion.get("expand_radius", 1),
        )
