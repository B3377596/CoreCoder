"""Recovery and retry system for task orchestration.

Handles:
- Retry decisions (uses node.retry_policy for config)
- Rollback hooks (clean up partial work before retry)
- Partial graph recovery (resume after interruption)
- Failure aggregation for replanning decisions

The recovery manager is invoked by the scheduler when a task fails.
It decides whether to retry, skip, or escalate to replanning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from corecoder.orchestration.dag.models import (
    TaskNode,
    TaskStatus,
    RetryPolicy,
    VerificationResult,
)


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
    It inspects the failure, consults the task's RetryPolicy, and returns a
    RecoveryAction.  The scheduler then carries out that action.

    Rollback hooks allow cleaning up partial work before retrying
    (e.g., deleting half-created files, reverting a partial migration).
    """

    def __init__(
        self,
        max_consecutive_failures: int = 5,
    ):
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
        1. Same error repeated → escalate immediately (not transient)
        2. Verifier says replan → replan
        3. Too many consecutive failures → abort (checked BEFORE verifier override)
        4. Retry budget remaining → retry
        5. Verifier override → retry (with hard cap at 2x max_retries)
        6. Otherwise → skip
        """
        max_r = node.retry_policy.max_retries

        # ---- Hard guard: same error repeating → not transient, stop ----
        prev_errors: list[str] = node.metadata.get("failure_history", [])
        if prev_errors and len(prev_errors) >= 3:
            # Check if the last N errors are all the same
            recent = []
            for e in prev_errors[-3:]:
                short = e.split("] ", 1)[-1] if "] " in e else e
                recent.append(short[:200])
            current_short = error[:200]
            if all(current_short[:100] in r or r in current_short[:100] for r in recent):
                self._consecutive_failures += 1
                return RecoveryAction(
                    action="skip",
                    task_id=node.id,
                    reason=f"Same error repeated {len(prev_errors)} times — "
                           f"not a transient failure. Last error: {error[:120]}",
                )

        # ---- Global failure guard (checked early) ----
        if self._consecutive_failures >= self.max_consecutive_failures:
            return RecoveryAction(
                action="abort",
                task_id=node.id,
                reason=f"Aborting after {self._consecutive_failures} consecutive failures",
            )

        # ---- Verifier-requested replan ----
        if verification and verification.should_replan:
            self._consecutive_failures = 0
            return RecoveryAction(
                action="replan",
                task_id=node.id,
                reason=verification.replan_hint or "Verifier requested replan",
            )

        # ---- Retry budget available ----
        if node.retry_policy.should_retry(node.retry_count, error):
            self._consecutive_failures += 1
            backoff = node.retry_policy.backoff_ms(node.retry_count + 1)
            return RecoveryAction(
                action="retry",
                task_id=node.id,
                reason=f"Retry {node.retry_count + 1}/{max_r}",
                backoff_ms=backoff,
            )

        # ---- Verifier override: ONE extra retry beyond budget ----
        if verification and verification.should_retry and node.retry_count < max_r * 2:
            self._consecutive_failures += 1
            backoff = node.retry_policy.backoff_ms(node.retry_count + 1)
            return RecoveryAction(
                action="retry",
                task_id=node.id,
                reason=f"Verifier override retry {node.retry_count + 1}/{max_r * 2}",
                backoff_ms=backoff,
            )

        # ---- Default: skip ----
        self._consecutive_failures += 1
        return RecoveryAction(
            action="skip",
            task_id=node.id,
            reason=f"Retry budget exhausted ({node.retry_count}/{max_r})",
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


class DefaultRetryPolicy(RetryPolicy):
    """Backward-compatible alias for the historical recovery API."""


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
