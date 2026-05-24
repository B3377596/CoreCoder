"""Structured observability for the orchestration layer.

Provides event-based logging for every state transition and decision
point in the DAG execution pipeline.  This is NOT a general-purpose
logging framework — it's a domain-specific observability layer that
emits typed events the user (or future monitoring systems) can consume.

Design decision: we use Python's `logging` module under the hood rather
than a custom event bus.  This keeps the dependency footprint zero and
lets users route events through their existing logging infrastructure.
Custom handlers can be added via `logger.addHandler()`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("corecoder.orchestration")


class EventType(str, Enum):
    """Typed events emitted by the orchestration pipeline."""

    # Graph lifecycle
    GRAPH_CREATED = "graph.created"
    GRAPH_LOADED = "graph.loaded"

    # Node lifecycle
    NODE_ADDED = "node.added"
    NODE_REMOVED = "node.removed"
    NODE_TRANSITION = "node.transition"
    NODE_RETRY = "node.retry"

    # Scheduling
    SCHEDULE_PICK = "schedule.pick"
    SCHEDULE_BLOCKED = "schedule.blocked"
    SCHEDULE_COMPLETE = "schedule.complete"

    # Execution
    EXECUTION_START = "execution.start"
    EXECUTION_END = "execution.end"
    EXECUTION_ERROR = "execution.error"

    # Verification
    VERIFY_START = "verify.start"
    VERIFY_PASS = "verify.pass"
    VERIFY_FAIL = "verify.fail"

    # Planner
    PLAN_START = "plan.start"
    PLAN_COMPLETE = "plan.complete"
    REPLAN_TRIGGERED = "replan.triggered"
    REPLAN_COMPLETE = "replan.complete"

    # Recovery
    RECOVERY_START = "recovery.start"
    RECOVERY_RETRY = "recovery.retry"
    RECOVERY_EXHAUSTED = "recovery.exhausted"


@dataclass
class TaskTransition:
    """A recorded state change for a single task node."""

    task_id: str
    task_title: str
    from_status: str
    to_status: str
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchestrationLogger:
    """Emits structured events for the orchestration pipeline.

    Usage:
        olog = OrchestrationLogger()
        olog.node_transition(task, "PENDING", "RUNNING", "scheduler picked it")
    """

    def __init__(self, name: str = "orchestration"):
        self._logger = logging.getLogger(f"corecoder.orchestration.{name}")
        self._transitions: list[TaskTransition] = []
        self._events: list[dict[str, Any]] = []  # full event log for replay
        self._start_time: float | None = None
        self._timers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        self._start_time = time.time()

    def elapsed_ms(self) -> float:
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000.0

    def start_timer(self, label: str) -> None:
        self._timers[label] = time.time()

    def stop_timer(self, label: str) -> float:
        """Return elapsed ms since start_timer was called for `label`."""
        start = self._timers.pop(label, None)
        if start is None:
            return 0.0
        return (time.time() - start) * 1000.0

    # ------------------------------------------------------------------
    # event emitters
    # ------------------------------------------------------------------

    def emit(self, event_type: EventType, **kwargs: Any) -> None:
        """Record and log a typed event."""
        event = {
            "type": event_type.value,
            "timestamp": time.time(),
            "elapsed_ms": self.elapsed_ms(),
            **kwargs,
        }
        self._events.append(event)
        self._logger.debug(
            "[%s] %s %s",
            event_type.value,
            kwargs.get("task_id", ""),
            kwargs.get("message", ""),
        )

    def node_transition(
        self,
        task_id: str,
        task_title: str,
        from_status: str,
        to_status: str,
        reason: str = "",
        **metadata: Any,
    ) -> None:
        """Record a task status transition."""
        transition = TaskTransition(
            task_id=task_id,
            task_title=task_title,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            metadata=metadata,
        )
        self._transitions.append(transition)

        self.emit(
            EventType.NODE_TRANSITION,
            task_id=task_id,
            task_title=task_title,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            **metadata,
        )

        self._logger.info(
            "[%s → %s] %s (%s)%s",
            from_status,
            to_status,
            task_title,
            task_id,
            f" — {reason}" if reason else "",
        )

    def retry_event(
        self, task_id: str, task_title: str, attempt: int, max_retries: int, error: str
    ) -> None:
        self.emit(
            EventType.NODE_RETRY,
            task_id=task_id,
            task_title=task_title,
            attempt=attempt,
            max_retries=max_retries,
            error=error,
        )
        self._logger.warning(
            "[retry %d/%d] %s (%s): %s",
            attempt,
            max_retries,
            task_title,
            task_id,
            error[:120],
        )

    def execution_timing(self, task_id: str, duration_ms: float, tokens: int) -> None:
        self._logger.info(
            "[timing] %s: %.0fms, %d tokens", task_id, duration_ms, tokens
        )

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of the orchestration run."""
        total_elapsed = self.elapsed_ms()
        transitions_by_task: dict[str, list[TaskTransition]] = {}
        for t in self._transitions:
            transitions_by_task.setdefault(t.task_id, []).append(t)

        return {
            "total_elapsed_ms": total_elapsed,
            "total_events": len(self._events),
            "total_transitions": len(self._transitions),
            "tasks_touched": len(transitions_by_task),
            "events_by_type": {},
            "transitions": [
                {
                    "task_id": t.task_id,
                    "task_title": t.task_title,
                    "from": t.from_status,
                    "to": t.to_status,
                    "reason": t.reason,
                    "elapsed_ms": (t.timestamp - self._start_time) * 1000.0
                    if self._start_time
                    else 0,
                }
                for t in self._transitions
            ],
        }
