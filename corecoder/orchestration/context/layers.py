"""Context layers — each layer is a factory producing typed ContextFragments.

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

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextSource,
    ContextType,
    ContextRequest,
    ExecutionState,
)


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
            if f.priority == 5:  # Unset — use layer default
                f.priority = self.default_priority
            if self.default_max_tokens > 0 and f.max_tokens == 0:
                f.max_tokens = self.default_max_tokens
        return fragments

    def _produce(self, request: ContextRequest) -> list[ContextFragment]:
        """Override in subclasses."""
        return []


# ===========================================================================
# System context — global instructions, role, capabilities
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
                token_count=_estimate_tokens(self._system_prompt),
            )
        ]


# ===========================================================================
# Task context — the current task node from the DAG
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
                token_count=_estimate_tokens(request.goal) + 3,
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
                token_count=_estimate_tokens(content),
                origin_task_id=request.task_id,
            ))

        return fragments


# ===========================================================================
# Working memory — in-flight execution state
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
                token_count=_estimate_tokens(content),
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
                token_count=_estimate_tokens(content),
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
                        parts.append(f"  - {key}: {', '.join(str(v) for v in value)}")
            content = "\n".join(parts)
            fragments.append(ContextFragment(
                source=self.source,
                type=ContextType.ARTIFACT,
                content=content,
                priority=7,
                relevance_score=0.85,
                token_count=_estimate_tokens(content),
            ))

        return fragments


# ===========================================================================
# Failure memory — past errors to avoid repeating
# ===========================================================================

class FailureMemoryContextLayer(ContextLayer):
    """Produces context about past failures and their root causes."""

    source = ContextSource.FAILURE_MEMORY
    default_priority = 8  # High priority — avoiding repeat failures is critical

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
                token_count=_estimate_tokens(content),
                ttl=300.0,  # Expire after 5 minutes
            ))

        return fragments


# ===========================================================================
# Constraint context — hard rules and limits
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
                token_count=_estimate_tokens(content),
            ))
        return fragments


# ===========================================================================
# Token estimation helper (shared)
# ===========================================================================

def _estimate_tokens(text: str) -> int:
    """Fast token count estimate — ~4 chars per token for code/mixed text."""
    if not text:
        return 0
    return max(1, len(text) // 3)
