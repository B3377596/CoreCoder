"""Query planner — converts TaskIntent into a structured RetrievalQuery.

Two-level planning:
  UNDERSTANDING → architecture overview files, entrypoints, structure
  EXECUTION     → symbol routing, dependency expansion, task-specific files

The planner bridges the gap between "the user asked about the project"
and "find entrypoints, architecture files, and structural overviews."

Design: lightweight symbolic reasoning.  No LLM calls.
"""

from __future__ import annotations

from typing import TypedDict

from corecoder.orchestration.retrieval.models import (
    TaskIntent,
    RetrievalQuery,
)


class TypeExpansion(TypedDict):
    extra_concepts: list[str]
    extra_files: list[str]
    expand_radius: int


class RetrievalQueryPlanner:
    """Converts TaskIntent → RetrievalQuery with mode-aware expansion.

    Usage:
        planner = RetrievalQueryPlanner()
        query = planner.plan(intent)
    """

    # ---- Task-type-specific expansions (EXECUTION family) ----
    _TYPE_EXPANSIONS: dict[str, TypeExpansion] = {
        "bug_fix": {
            "extra_concepts": ["error", "exception", "validation", "edge case"],
            "extra_files": [],
            "expand_radius": 2,
        },
        "feature_integration": {
            "extra_concepts": ["dispatch", "route", "entrypoint", "interface",
                               "integration", "wiring"],
            "extra_files": ["main.py", "cli.py", "app.py", "__init__.py"],
            "expand_radius": 2,
        },
        "feature_addition": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 1,
        },
        "cli_change": {
            "extra_concepts": ["argparse", "click", "typer", "command", "argument",
                               "flag", "option", "terminal", "console"],
            "extra_files": ["main.py", "cli.py", "app.py", "run.py", "__main__.py"],
            "expand_radius": 1,
        },
        "refactor": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 3,
        },
        "rename": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 2,
        },
        "dependency_change": {
            "extra_concepts": ["import", "dependency", "package", "install"],
            "extra_files": ["pyproject.toml", "setup.py", "requirements.txt",
                            "setup.cfg", "__init__.py"],
            "expand_radius": 1,
        },
        "test_addition": {
            "extra_concepts": ["test", "assert", "mock", "fixture", "coverage"],
            "extra_files": [],
            "expand_radius": 1,
        },
        "documentation": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 0,
        },
        "unknown": {
            "extra_concepts": [],
            "extra_files": [],
            "expand_radius": 1,
        },
    }

    # ---- Understanding-mode file priorities ----
    # For understanding queries, prioritize files that reveal architecture:
    # entrypoints, top-level modules, package inits, config files.
    _UNDERSTANDING_PRIORITY_FILES: list[str] = [
        # Entrypoints
        "main.py", "cli.py", "app.py", "run.py", "__main__.py", "server.py",
        # Project metadata
        "pyproject.toml", "setup.py", "setup.cfg", "Cargo.toml",
        "package.json", "go.mod", "Makefile",
        # Package structure
        "__init__.py",
        # Documentation
        "README.md", "README_CN.md", "README.rst",
    ]

    # Understanding concepts → file categories to prioritize
    _UNDERSTANDING_CONCEPT_FILE_MAP: dict[str, list[str]] = {
        "architecture": ["__init__.py", "main.py", "cli.py", "app.py"],
        "overview": ["main.py", "cli.py", "app.py", "__init__.py"],
        "entrypoint": ["main.py", "cli.py", "app.py", "run.py", "__main__.py", "server.py"],
        "capabilities": ["main.py", "cli.py", "__init__.py"],
        "components": ["__init__.py"],
        "modules": ["__init__.py"],
        "purpose": ["main.py", "cli.py", "pyproject.toml", "setup.py"],
        "execution_flow": ["main.py", "cli.py", "app.py", "__main__.py"],
        "structure": ["__init__.py"],
    }

    def plan(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan a retrieval query from task intent.

        Routes to mode-specific planning based on the intent family.
        """
        if intent.family == "understanding":
            return self._plan_understanding(intent)
        elif intent.family == "navigation":
            return self._plan_navigation(intent)
        elif intent.family == "explanation":
            return self._plan_explanation(intent)
        elif intent.family == "planning":
            return self._plan_planning(intent)
        else:
            return self._plan_execution(intent)

    # ------------------------------------------------------------------
    # Mode-specific planners
    # ------------------------------------------------------------------

    def _plan_understanding(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan for understanding queries.

        Prioritizes architecture-revealing files: entrypoints, top-level
        modules, package inits.  Does NOT do symbol routing — the user
        wants project shape, not symbol locations.
        """
        # Map concepts to priority files
        likely_files: list[str] = []
        for concept in intent.concepts:
            for mapped in self._UNDERSTANDING_CONCEPT_FILE_MAP.get(concept, []):
                if mapped not in likely_files:
                    likely_files.append(mapped)

        # Always include the base understanding priority files
        for f in self._UNDERSTANDING_PRIORITY_FILES:
            if f not in likely_files:
                likely_files.append(f)

        return RetrievalQuery(
            symbols=[],  # Understanding queries don't use symbol routing
            concepts=intent.concepts,
            likely_files=likely_files,
            task_type=intent.type,
            expand_dependencies=True,
            dependency_radius=2,  # Wider radius for architecture understanding
        )

    def _plan_execution(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan for execution (task-oriented) queries.

        Uses the existing type-specific expansions — symbol routing,
        task-type file preferences, dependency expansion.
        """
        expansion = self._TYPE_EXPANSIONS.get(
            intent.type, self._TYPE_EXPANSIONS["unknown"]
        )

        concepts = list(dict.fromkeys(intent.concepts + expansion["extra_concepts"]))
        likely_files = list(dict.fromkeys(intent.affected_files + expansion["extra_files"]))

        return RetrievalQuery(
            symbols=intent.symbols,
            concepts=concepts,
            likely_files=likely_files,
            task_type=intent.type,
            expand_dependencies=expansion["expand_radius"] > 0,
            dependency_radius=expansion["expand_radius"],
        )

    def _plan_navigation(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan for navigation queries ("where is X?").

        Symbols are the primary signal; concepts secondary.
        """
        return RetrievalQuery(
            symbols=intent.symbols,
            concepts=intent.concepts,
            likely_files=intent.affected_files,
            task_type="navigation",
            expand_dependencies=False,
            dependency_radius=0,
        )

    def _plan_explanation(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan for explanation queries ("how does X work?").

        Deep dive: broad dependency radius, all related files.
        """
        return RetrievalQuery(
            symbols=intent.symbols,
            concepts=intent.concepts,
            likely_files=intent.affected_files,
            task_type="deep_dive",
            expand_dependencies=True,
            dependency_radius=3,
        )

    def _plan_planning(self, intent: TaskIntent) -> RetrievalQuery:
        """Plan for planning queries ("how should I build X?").

        Broad overview: entrypoints + architecture + capabilities.
        """
        return RetrievalQuery(
            symbols=intent.symbols,
            concepts=intent.concepts + ["architecture", "overview", "entrypoint"],
            likely_files=self._UNDERSTANDING_PRIORITY_FILES,
            task_type="planning",
            expand_dependencies=True,
            dependency_radius=2,
        )
