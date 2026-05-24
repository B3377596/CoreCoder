"""Recovery and retry system for task orchestration.

Handles:
- Retry policies (exponential backoff, max attempts)
- Rollback hooks (clean up partial work before retry)
- Partial graph recovery (resume after interruption)
- Failure aggregation for replanning decisions

The recovery manager is invoked by the scheduler when a task fails.
It decides whether to retry, skip, or escalate to replanning.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from corecoder.orchestration.models import (
    TaskNode,
    TaskStatus,
    ExecutionResult,
    VerificationResult,
    RetryPolicy,
)


class BaseRetryPolicy(ABC):
    """Abstract retry policy — decides if and when to retry a failed task."""

    @abstractmethod
    def should_retry(self, node: TaskNode, error: str) -> bool:
        """Return True if the task should be retried."""

    @abstractmethod
    def backoff_ms(self, attempt: int) -> float:
        """Return the backoff delay in milliseconds for this attempt."""


class DefaultRetryPolicy(BaseRetryPolicy):
    """Standard exponential backoff with max retries.

    Delay = min(base * 2^attempt, max_delay)
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base_ms: float = 1000.0,
        backoff_max_ms: float = 60_000.0,
        retry_on: tuple[str, ...] = (),
    ):
        self.max_retries = max_retries
        self.backoff_base_ms = backoff_base_ms
        self.backoff_max_ms = backoff_max_ms
        self.retry_on = retry_on

    def should_retry(self, node: TaskNode, error: str) -> bool:
        if node.retry_count >= self.max_retries:
            return False
        if self.retry_on:
            return any(pattern in error for pattern in self.retry_on)
        return True

    def backoff_ms(self, attempt: int) -> float:
        delay = self.backoff_base_ms * (2 ** attempt)
        return min(delay, self.backoff_max_ms)


@dataclass
class RecoveryAction:
    """What the recovery manager decided to do about a failed task."""

    action: str  # "retry", "skip", "replan", "abort"
    task_id: str
    reason: str = ""
    backoff_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """Decides what to do when a task fails.

    The recovery manager is a decision-making layer, NOT an execution layer.
    It inspects the failure, consults the retry policy, and returns a
    RecoveryAction.  The scheduler then carries out that action.

    Rollback hooks allow cleaning up partial work before retrying
    (e.g., deleting half-created files, reverting a partial migration).
    """

    def __init__(
        self,
        retry_policy: BaseRetryPolicy | None = None,
        max_consecutive_failures: int = 5,
    ):
        self._policy = retry_policy or DefaultRetryPolicy()
        self._rollback_hooks: dict[str, list[Callable[[TaskNode], None]]] = {}
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # rollback hooks
    # ------------------------------------------------------------------

    def register_rollback(self, task_id: str, hook: Callable[[TaskNode], None]) -> None:
        """Register a cleanup function to run before retrying a task.

        Hooks receive the task node so they can inspect its artifacts
        and metadata to know what to clean up.
        """
        self._rollback_hooks.setdefault(task_id, []).append(hook)

    def run_rollbacks(self, node: TaskNode) -> None:
        """Execute all registered rollback hooks for a task."""
        for hook in self._rollback_hooks.get(node.id, []):
            try:
                hook(node)
            except Exception as e:
                # Rollback failures are logged but don't prevent retry
                node.metadata.setdefault("rollback_errors", []).append(str(e))

    # ------------------------------------------------------------------
    # recovery decisions
    # ------------------------------------------------------------------

    def decide(
        self,
        node: TaskNode,
        error: str,
        verification: VerificationResult | None = None,
    ) -> RecoveryAction:
        """Decide what to do about a failed task.

        Decision priority:
        1. If verifier says replan → replan
        2. If retry budget remaining → retry
        3. If verifier says retry even without budget → retry (override)
        4. If too many consecutive failures → abort
        5. Otherwise → skip
        """
        # Verifier-requested replan takes precedence
        if verification and verification.should_replan:
            self._consecutive_failures = 0
            return RecoveryAction(
                action="replan",
                task_id=node.id,
                reason=verification.replan_hint or "Verifier requested replan",
            )

        # Check retry budget
        if self._policy.should_retry(node, error):
            self._consecutive_failures += 1
            backoff = self._policy.backoff_ms(node.retry_count + 1)
            return RecoveryAction(
                action="retry",
                task_id=node.id,
                reason=f"Retry {node.retry_count + 1}/{node.retry_policy.max_retries}",
                backoff_ms=backoff,
            )

        # Verifier override: retry even if policy says no
        if verification and verification.should_retry:
            self._consecutive_failures += 1
            return RecoveryAction(
                action="retry",
                task_id=node.id,
                reason="Verifier override: should retry",
                backoff_ms=self._policy.backoff_ms(node.retry_count + 1),
            )

        # Global failure guard
        if self._consecutive_failures >= self.max_consecutive_failures:
            return RecoveryAction(
                action="abort",
                task_id=node.id,
                reason=f"Aborting after {self._consecutive_failures} consecutive failures",
            )

        # Default: skip this task, mark dependents as blocked
        self._consecutive_failures += 1
        return RecoveryAction(
            action="skip",
            task_id=node.id,
            reason=f"Retry budget exhausted ({node.retry_count}/{node.retry_policy.max_retries})",
        )

    def reset_consecutive_failures(self) -> None:
        """Reset the consecutive failure counter (call after a successful task)."""
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # graph recovery (resume interrupted execution)
    # ------------------------------------------------------------------

    def prepare_retry(self, node: TaskNode) -> TaskNode:
        """Prepare a task node for retry: increment counter, reset status."""
        node.retry_count += 1
        node.transition_to(TaskStatus.PENDING)
        # Record the failure in history so MemoryInjector can surface it
        if node.error:
            node.metadata.setdefault("failure_history", []).append(
                f"[Attempt {node.retry_count}] {node.error}"
            )
        node.error = None
        node.result = None
        node.verification = None
        return node

    async def wait_backoff(self, backoff_ms: float) -> None:
        """Wait for the backoff period before retrying."""
        if backoff_ms > 0:
            await asyncio.sleep(backoff_ms / 1000.0)


def resume_graph_state(graph, from_dict: dict) -> list[str]:
    """Recover a graph from persisted state after interruption.

    Identifies tasks that were RUNNING at interruption time and resets
    them to PENDING so they can be re-executed.

    Returns the list of task IDs that were reset.
    """
    reset_ids: list[str] = []
    for node_data in from_dict.get("nodes", []):
        node_id = node_data["id"]
        status = node_data.get("status", "pending")
        if status == "running":
            node = graph.get_node(node_id)
            if node is not None:
                node.transition_to(TaskStatus.PENDING)
                node.error = "Interrupted — reset for recovery"
                reset_ids.append(node_id)
    return reset_ids
