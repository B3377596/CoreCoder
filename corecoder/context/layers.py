"""Context layers  each layer is a factory producing typed ContextFragments.

Each layer:
- Has its own retrieval logic
- Own token budget sub-allocation
- Own compression policy
- Own default priority

The ContextOrchestrator invokes each layer with a ContextRequest and
collects the resulting fragments for the pipeline.
"""

from __future__ import annotations

from typing import Any

from corecoder.context.models import (
    ContextFragment,
    ContextSource,
    ContextType,
    ContextRequest,
    ExecutionState,
)
from corecoder.context.compression import count_tokens


# ===========================================================================
# Base layer
# ===========================================================================

class ContextLayer:
    """Base class for all context layers.

    Subclasses implement _produce() to generate fragments for a request.
    """

    source: ContextSource
    default_priority: int = 5
    default_max_tokens: int = 0  # 0 = no per-fragment limit

    def produce(self, request: ContextRequest) -> list[ContextFragment]:
        """Generate fragments for this layer.  Template method."""
        fragments = self._produce(request)
        # Stamp source and priority on every fragment
        for f in fragments:
            f.source = self.source
            if f.priority == 5:  # Unset  use layer default
                f.priority = self.default_priority
            if self.default_max_tokens > 0 and f.max_tokens == 0:
                f.max_tokens = self.default_max_tokens
        return fragments

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        """Override in subclasses."""
        return []


# ===========================================================================
# System context  global instructions, role, capabilities
# ===========================================================================

class SystemContextLayer(ContextLayer):
    """Produces system-level context: the agent's role, capabilities, and rules."""

    source = ContextSource.SYSTEM
    default_priority = 10  # System context is always kept

    def __init__(self, system_prompt: str = ""):
        self._system_prompt = system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        if not self._system_prompt:
            return []
        return [
            ContextFragment(
                source=self.source,
                type=ContextType.INSTRUCTION,
                content=self._system_prompt,
                priority=10,
                relevance_score=1.0,
                token_count=count_tokens(self._system_prompt),
            )
        ]


# ===========================================================================
# Task context  the current task node from the DAG
# ===========================================================================

class TaskContextLayer(ContextLayer):
    """Produces context about the current task being executed."""

    source = ContextSource.TASK
    default_priority = 9

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        fragments: list[ContextFragment] = []

        # Overall goal
        if request.goal:
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.INSTRUCTION,
                content=f"## Goal\n{request.goal}",
                priority=10,
                relevance_score=1.0,
                token_count=count_tokens(request.goal) + 3,
            ))

        # Current task
        if request.task_title:
            content = f"## Current Task: {request.task_title}\n{request.task_description}"
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.INSTRUCTION,
                content=content,
                priority=9,
                relevance_score=1.0,
                token_count=count_tokens(content),
                origin_task_id=request.task_id,
            ))

        return fragments


# ===========================================================================
# Working memory  in-flight execution state
# ===========================================================================

class WorkingMemoryContextLayer(ContextLayer):
    """Produces working memory context: assumptions, open questions, discoveries."""

    source = ContextSource.WORKING_MEMORY
    default_priority = 7

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        fragments: list[ContextFragment] = []

        # Assumptions
        if request.assumptions:
            content = "## Assumptions\n" + "\n".join(f"- {a}" for a in request.assumptions)
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.CONSTRAINT,
                content=content,
                priority=6,
                relevance_score=0.8,
                token_count=count_tokens(content),
            ))

        # Constraints
        if request.constraints:
            content = "## Constraints\n" + "\n".join(f"- {c}" for c in request.constraints)
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.CONSTRAINT,
                content=content,
                priority=8,
                relevance_score=0.9,
                token_count=count_tokens(content),
            ))

        # Completed upstream work
        if request.completed_artifact_map:
            parts = ["## Completed Prerequisites"]
            for task_id, art in request.completed_artifact_map.items():
                desc = art.get("description", task_id)
                parts.append(f"- {desc}")
                for key, value in art.items():
                    if key == "description":
                        continue
                    if isinstance(value, list):
                        parts.append(f"  - {key}: {', '.join(str(v) for v in value[:8])}")
            content = "\n".join(parts)
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.ARTIFACT,
                content=content,
                priority=7,
                relevance_score=0.85,
                token_count=count_tokens(content),
            ))

        return fragments


# ===========================================================================
# Failure memory  past errors to avoid repeating
# ===========================================================================

class FailureMemoryContextLayer(ContextLayer):
    """Produces context about past failures and their root causes."""

    source = ContextSource.FAILURE_MEMORY
    default_priority = 8  # High priority  avoiding repeat failures is critical

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        fragments: list[ContextFragment] = []

        if request.recent_errors:
            content = "## Recent Failures (do NOT repeat)\n"
            for err in request.recent_errors[-5:]:  # Last 5 only
                content += f"- {err[:300]}\n"
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.ERROR,
                content=content,
                priority=8,
                relevance_score=0.95,
                token_count=count_tokens(content),
                ttl=300.0,  # Expire after 5 minutes
            ))

        return fragments


# ===========================================================================
# Constraint context  hard rules and limits
# ===========================================================================

class ConstraintContextLayer(ContextLayer):
    """Produces hard constraints that must not be violated."""

    source = ContextSource.CONSTRAINT
    default_priority = 9

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        fragments: list[ContextFragment] = []
        if request.constraints:
            content = "## Hard Constraints\n" + "\n".join(
                f"- {c}" for c in request.constraints
            )
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.CONSTRAINT,
                content=content,
                priority=9,
                relevance_score=1.0,
                token_count=count_tokens(content),
            ))
        return fragments


# ===========================================================================
# Execution policy  task contract: boundaries, stop conditions, anti-redundancy
# ===========================================================================

class ExecutionPolicyContextLayer(ContextLayer):
    """Produces the task contract  hard execution boundaries.

    This is the "action gating" layer.  Without it, the agent treats task
    descriptions as suggestions and freely does downstream work.
    """

    source = ContextSource.CONSTRAINT
    default_priority = 10  # Highest priority  contract must always be visible

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        fragments: list[ContextFragment] = []
        title = request.task_title.lower()
        desc = request.task_description.lower()
        meta = request.metadata

        # ---- Downstream tasks (what NOT to do) ----
        downstream_ids = meta.get("downstream_tasks", [])
        if downstream_ids:
            lines = ["**DOWNSTREAM TASKS (do NOT do these):**"]
            for d in downstream_ids:
                lines.append(f"  - FORBIDDEN: {d}")
            lines.append(
                "\nIf you find yourself doing any of the above, STOP immediately. "
                "Those belong to later tasks in the pipeline."
            )
            fragments.append(ContextFragment(
                source=self.source, type=ContextType.CONSTRAINT,
                content="\n".join(lines), priority=10, relevance_score=1.0,
                token_count=count_tokens("\n".join(lines)),
            ))

        # ---- Allowed / Forbidden ----
        # Prefer planner-generated bounds (LLM). Fall back to keyword heuristics.
        allowed = meta.get("task_allowed")
        forbidden = meta.get("task_forbidden")
        if allowed is None and forbidden is None:
            allowed, forbidden = self._derive_bounds(title, desc)

        if allowed or forbidden:
            lines = []
            if allowed:
                lines.append(f"**ALLOWED**: {', '.join(allowed)}")
            if forbidden:
                lines.append(f"**FORBIDDEN**: {', '.join(forbidden)}")
            fragments.append(ContextFragment(
                source=self.source, type=ContextType.CONSTRAINT,
                content="\n".join(lines), priority=10, relevance_score=1.0,
                token_count=count_tokens("\n".join(lines)),
            ))

        # ---- Stop conditions ----
        # Prefer planner-generated stop condition. Fall back to keywords.
        stop = meta.get("task_stop_when", "") or self._derive_stop(title, desc)
        if stop:
            fragments.append(ContextFragment(
                source=self.source, type=ContextType.INSTRUCTION,
                content=f"**STOP WHEN**: {stop}. Then STOP. Do not do extra work.",
                priority=10, relevance_score=1.0,
                token_count=count_tokens(stop) + 20,
            ))

        return fragments
    def _derive_bounds(self, title: str, desc: str) -> tuple[list[str], list[str]]:
        return derive_task_bounds(title, desc)

    def _derive_stop(self, title: str, desc: str) -> str:
        return derive_task_stop(title, desc)


# ===========================================================================
# Shared keyword heuristics  used by both ExecutionPolicyContextLayer
# (fallback) and LLMPlanner (auto-fill missing bounds after parsing).
# ===========================================================================


def derive_task_bounds(title: str, desc: str) -> tuple[list[str], list[str]]:
    """Derive allowed and forbidden actions from task keywords."""
    text = (title + " " + desc).lower()

    if any(kw in text for kw in
           ("install", "dependency", "dependencies", "package", "uv add", "pip install")):
        return (
            ["run package manager commands (uv add, pip install, etc.)",
             "edit pyproject.toml/requirements.txt to declare dependencies"],
            ["write application code", "create .py files", "write server files",
             "start servers", "create test files", "implement features"],
        )
    if any(kw in text for kw in
           ("init", "venv", "virtual environment", "setup", "create project", "uv init")):
        return (
            ["run 'uv init' or 'uv venv'", "create config files"],
            ["write application code", "create test files", "install testing packages",
             "run tests", "implement features or algorithms"],
        )
    if any(kw in text for kw in
           ("implement", "write", "create", "code", "logic", "function")):
        return (
            ["write the specified code files", "run once with basic input to verify"],
            ["initialize package managers", "create virtual environments",
             "create test files", "install testing packages", "run test suites"],
        )
    if any(kw in text for kw in
           ("ui", "interface", "cli", "command line", "entry point")):
        return (
            ["create or modify the CLI/UI entry point", "wire existing modules"],
            ["re-initialize the project", "re-create existing code files", "create tests"],
        )
    return ([], [])


def derive_task_stop(title: str, desc: str) -> str:
    """Derive stop conditions from task keywords."""
    text = (title + " " + desc).lower()
    if any(kw in text for kw in
           ("install", "dependency", "dependencies", "uv add", "pip install")):
        return "the dependency appears in pyproject.toml or the package manager reports success"
    if any(kw in text for kw in
           ("init", "venv", "virtual environment", "setup", "create project")):
        return ".venv/ or pyproject.toml exists"
    if any(kw in text for kw in
           ("implement", "write", "create", "code", "logic")):
        return "the code file exists and runs correctly on one basic input"
    if any(kw in text for kw in
           ("ui", "interface", "cli", "command line")):
        return "the entry point works end-to-end"
    return ""


