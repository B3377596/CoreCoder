"""Executor — wraps the existing ReAct agent loop for task execution.

Each task node in the DAG is executed by the existing Agent.chat() loop.
This module provides the bridge between orchestration-level task nodes
and the agent-level conversation loop.

Design principle: the executor is a THIN wrapper.  It does NOT reimplement
the agent loop.  It:
1. Receives a TaskNode + WorkingMemory from the scheduler
2. Builds a prompt from the working memory
3. Delegates to Agent.chat()
4. Captures the result as an ExecutionResult
5. Returns control to the scheduler

The executor owns NO mutable state.  It's a pure function from
(task, memory, agent) → ExecutionResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from corecoder.orchestration.models import TaskNode, ExecutionResult
from corecoder.orchestration.memory import WorkingMemory


@dataclass
class TaskContext:
    """Everything the executor needs to run one task.

    Bundled into a single object so the scheduler can pass it cleanly
    without positional-argument sprawl.
    """

    task: TaskNode
    memory: WorkingMemory
    working_dir: str = "."
    # Hook for preprocessing the messages before sending to the agent
    preprocess: Callable[[list[dict]], list[dict]] | None = None


# Type alias for the agent callable — this is what we wrap
# AgentCallable = Callable[[str], Awaitable[str]]
AgentCallable = Callable[..., Awaitable[str]]


class MaxRoundsExceededError(Exception):
    """Raised when a single orchestrated task exceeds its max tool-call rounds."""
    pass


class Executor:
    """Executes a single task node using the existing Agent ReAct loop.

    Usage:
        executor = Executor(agent_chat_fn=agent.chat)
        result = await executor.execute(task_context)
    """

    def __init__(
        self,
        agent_chat_fn: AgentCallable | None = None,
        default_timeout_ms: float = 300_000.0,  # 5 minutes per task
        max_rounds_per_task: int = 20,
    ):
        self._chat = agent_chat_fn
        self._timeout_ms = default_timeout_ms
        self._max_rounds = max_rounds_per_task

        # Optional callbacks — set by the scheduler for observability
        self._on_tool: Callable[[str, dict], None] | None = None
        self._on_token: Callable[[str], None] | None = None

    def set_agent(self, agent_chat_fn: AgentCallable) -> None:
        """Inject the agent chat function after construction."""
        self._chat = agent_chat_fn

    def set_callbacks(
        self,
        on_tool: Callable[[str, dict], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> None:
        """Register callbacks for tool-call observability."""
        self._on_tool = on_tool
        self._on_token = on_token

    async def execute(self, ctx: TaskContext) -> ExecutionResult:
        """Execute a single task node.

        1. Build the user prompt from working memory
        2. Invoke the agent loop (with tool-call callback)
        3. Collect timing and token info
        4. Return structured ExecutionResult

        Each orchestrated task gets a separate tool-call round limit
        (default 20) to prevent infinite loops in stuck agents.
        """
        if self._chat is None:
            raise RuntimeError("Executor has no agent.  Call set_agent() first.")

        task = ctx.task
        task.status = "running"
        task.touch()

        user_message = self._build_task_prompt(ctx)

        start = time.time()
        round_count = [0]  # mutable counter for closure

        def _on_tool(name: str, kwargs: dict) -> None:
            round_count[0] += 1
            if round_count[0] > self._max_rounds:
                raise MaxRoundsExceededError(
                    f"Task '{task.title}' exceeded max rounds "
                    f"({self._max_rounds} tool calls)"
                )
            if self._on_tool:
                self._on_tool(name, kwargs)

        def _on_token(tok: str) -> None:
            if self._on_token:
                self._on_token(tok)

        try:
            # Delegate to the existing agent loop — pass callbacks so tool
            # invocations are visible to the UI and round-counted.
            # Agent.chat() signature is: chat(user_input, on_token=None, on_tool=None)
            output = await self._chat(user_message, on_token=_on_token, on_tool=_on_tool)
            duration_ms = (time.time() - start) * 1000.0

            artifacts = self._extract_artifacts(ctx)

            result = ExecutionResult(
                success=True,
                output=output,
                tool_calls_made=round_count[0],
                duration_ms=duration_ms,
                artifacts=artifacts,
                metadata={
                    "task_id": task.id,
                    "task_title": task.title,
                    "attempt": task.retry_count + 1,
                },
            )
        except MaxRoundsExceededError as e:
            duration_ms = (time.time() - start) * 1000.0
            result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                tool_calls_made=round_count[0],
                duration_ms=duration_ms,
                metadata={
                    "task_id": task.id,
                    "task_title": task.title,
                    "attempt": task.retry_count + 1,
                    "exception_type": "MaxRoundsExceededError",
                },
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000.0
            result = ExecutionResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {e}",
                tool_calls_made=round_count[0],
                duration_ms=duration_ms,
                metadata={
                    "task_id": task.id,
                    "task_title": task.title,
                    "attempt": task.retry_count + 1,
                    "exception_type": type(e).__name__,
                },
            )

        return result

    def _build_task_prompt(self, ctx: TaskContext) -> str:
        """Build the user prompt that drives the agent for this task.

        The prompt combines:
        - The overall goal context from working memory
        - The specific task instructions
        - Information about completed dependencies
        - Known constraints and previous failures
        """
        memory = ctx.memory

        parts: list[str] = []

        # Main instruction
        parts.append(f"## Task: {memory.current_task_title}\n\n{memory.current_task_description}")

        # Context from the overall goal
        if memory.current_goal:
            parts.append(f"\n## Overall Goal\n{memory.current_goal}")

        # Completed upstream work
        if memory.completed_artifacts:
            parts.append("\n## Completed Prerequisites")
            for tid, art in memory.completed_artifacts.items():
                desc = art.get("description", tid)
                parts.append(f"- {desc}")
                files = art.get("files", art.get("expected_files", []))
                if files:
                    for f in files:
                        parts.append(f"  - Created/modified: {f}")

        # Known constraints
        if memory.known_constraints:
            parts.append("\n## Constraints (must follow)")
            for c in memory.known_constraints:
                parts.append(f"- {c}")

        # Previous failures to avoid
        if memory.recent_failures:
            parts.append("\n## Previous Failures (do NOT repeat)")
            for f in memory.recent_failures:
                parts.append(f"- {f}")

        # Assumptions
        if memory.assumptions:
            parts.append("\n## Assumptions")
            for a in memory.assumptions:
                parts.append(f"- {a}")

        # Notes
        if memory.notes:
            parts.append(f"\n## Notes\n{memory.notes}")

        # Output format instruction
        parts.append(
            "\n\nWhen you have completed this task, summarize what you did "
            "and what files you changed.  Be specific about file paths."
        )

        return "\n".join(parts)

    def _extract_artifacts(self, ctx: TaskContext) -> dict[str, Any]:
        """Extract artifact information from task metadata.

        This is a best-effort extraction — the real artifacts are the
        file changes tracked by shadow git.  We store the expected files
        list so downstream tasks know what to look for.
        """
        artifacts: dict[str, Any] = {}
        verification = ctx.task.metadata.get("verification", {})
        if isinstance(verification, dict):
            expected = verification.get("expected_files", [])
            if expected:
                artifacts["expected_files"] = expected
        return artifacts
