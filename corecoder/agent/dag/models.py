"""Core data models for the task orchestration layer.

All mutable state flows through these typed containers.  We use dataclasses
rather than Pydantic to avoid adding a dependency ? the orchestration layer
should be zero-dependency beyond what corecoder already requires.
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class TaskStatus(str, Enum):
    """Atomic states a task node can occupy.

    The state machine is:
        PENDING ? READY ? RUNNING ? SUCCESS
                  ?          ?
                  ?       FAILED ? (retry) ? READY
                  ?          ?
                  ?       BLOCKED
                  ?
              BLOCKED ? READY  (when dependencies resolve after replan)

    SKIPPED: set manually when a task becomes irrelevant (e.g. a parent
    failed and the user decides to skip downstream tasks).
    """

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        """True if this state ends the task's lifecycle (no further transitions)."""
        return self in (TaskStatus.SUCCESS, TaskStatus.SKIPPED)

    @property
    def is_recoverable(self) -> bool:
        """True if the task can be retried from this state."""
        return self in (TaskStatus.FAILED,)


@dataclass
class ExecutionResult:
    """The output of a single task execution attempt."""

    success: bool
    output: str = ""
    error: str = ""
    tool_calls_made: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Output from the verifier layer after task execution."""

    passed: bool
    checks_run: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    # hints for the scheduler about what to do next
    should_retry: bool = False
    should_replan: bool = False
    # optional structured data for replanning
    replan_hint: str = ""


@dataclass
class RetryPolicy:
    """Configuration for how a task should be retried on failure."""

    max_retries: int = 3
    backoff_base_ms: float = 1000.0  # exponential backoff: base * 2^attempt
    backoff_max_ms: float = 60_000.0
    retry_on: tuple[str, ...] = ()  # error substrings that trigger retry; empty = all
    reset_context_on_retry: bool = False  # whether to wipe messages before retry

    def should_retry(self, retry_count: int, error: str = "") -> bool:
        """Return True if the task should be retried given the attempt count."""
        if retry_count >= self.max_retries:
            return False
        if self.retry_on:
            return any(pattern in error for pattern in self.retry_on)
        return True

    def backoff_ms(self, attempt: int) -> float:
        """Return the backoff delay in milliseconds for this attempt."""
        delay = self.backoff_base_ms * (2 ** attempt)
        return min(delay, self.backoff_max_ms)


@dataclass
class TaskNode:
    """A single node in the task DAG.

    Each node represents one unit of work.  Dependencies are tracked by ID
    reference ? the graph object owns the resolution logic.

    Design note: `artifacts` and `metadata` are deliberately untyped dicts.
    They carry opaque data between nodes (file paths, test results, generated
    code) without the orchestration layer needing to understand the contents.
    Type safety at this boundary would add ceremony without value.
    """

    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0  # higher = more urgent; scheduler picks highest first
    retry_count: int = 0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # execution results (set after each attempt)
    result: ExecutionResult | None = None
    verification: VerificationResult | None = None
    error: str | None = None

    # opaque data carriers between task nodes
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = time.time()

    def transition_to(self, new_status: TaskStatus | str) -> None:
        """Atomically move to a new status with timestamp update.

        Accepts both TaskStatus enum values and string names
        (e.g. ``TaskStatus.RUNNING`` or ``"running"``).
        """
        if isinstance(new_status, str):
            new_status = TaskStatus(new_status)
        self.status = new_status
        self.touch()

    def record_result(self, result: ExecutionResult) -> None:
        """Store execution output and update status accordingly."""
        self.result = result
        if result.success:
            self.transition_to(TaskStatus.SUCCESS)
        else:
            self.error = result.error
            if self.retry_count < self.retry_policy.max_retries:
                self.transition_to(TaskStatus.FAILED)
            else:
                self.transition_to(TaskStatus.FAILED)

    def can_retry(self) -> bool:
        """Check whether this task still has retry budget."""
        return self.retry_count < self.retry_policy.max_retries

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence.  Not full fidelity ? enough for recovery."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.retry_policy.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "result": {
                "success": self.result.success,
                "output": self.result.output,
                "error": self.result.error,
                "tool_calls_made": self.result.tool_calls_made,
                "tokens_used": self.result.tokens_used,
                "duration_ms": self.result.duration_ms,
            }
            if self.result
            else None,
            "verification": {
                "passed": self.verification.passed,
                "checks_run": self.verification.checks_run,
                "failures": self.verification.failures,
            }
            if self.verification
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskNode:
        """Restore a TaskNode from serialized state."""
        node = cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data["status"]),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", 0),
            retry_count=data.get("retry_count", 0),
            retry_policy=RetryPolicy(max_retries=data.get("max_retries", 3)),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            error=data.get("error"),
            artifacts=data.get("artifacts", {}),
            metadata=data.get("metadata", {}),
        )
        if data.get("result"):
            node.result = ExecutionResult(
                success=data["result"]["success"],
                output=data["result"].get("output", ""),
                error=data["result"].get("error", ""),
                tool_calls_made=data["result"].get("tool_calls_made", 0),
                tokens_used=data["result"].get("tokens_used", 0),
                duration_ms=data["result"].get("duration_ms", 0.0),
            )
        return node

    def __repr__(self) -> str:
        return f"TaskNode(id={self.id!r}, title={self.title!r}, status={self.status.value})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskNode):
            return NotImplemented
        return self.id == other.id
