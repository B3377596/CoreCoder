"""Scheduler — the execution engine for the task DAG.

The scheduler is the central coordinator.  It:
1. Queries the graph for ready tasks
2. Selects the next task to execute (by priority)
3. Builds working memory for the selected task
4. Delegates execution to the Executor
5. Runs verification on the result
6. Updates graph state based on outcome
7. Handles retries and failure propagation
8. Signals the orchestrator when done or when replanning is needed

The scheduler is designed for sequential execution initially.  The
_execute_single method is the extension point for parallel execution —
replace the await loop with asyncio.gather and a semaphore for
concurrency control.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from corecoder.orchestration.dag.models import TaskNode, TaskStatus, ExecutionResult
from corecoder.orchestration.dag.graph import TaskGraph
from corecoder.orchestration.dag.recovery import RecoveryManager, RecoveryAction
from corecoder.orchestration.engine.verifier import BaseVerifier, NoOpVerifier
from corecoder.orchestration.engine.executor import Executor, TaskContext
from corecoder.orchestration.dag.memory import MemoryInjector, WorkingMemory
from corecoder.orchestration.observability import OrchestrationLogger, EventType


class SchedulingDecision(str, Enum):
    """What the scheduler decided to do after processing a task."""

    CONTINUE = "continue"  # More tasks to run
    COMPLETE = "complete"  # All tasks done successfully
    FAILED = "failed"  # Unrecoverable failure
    REPLAN = "replan"  # Need to replan
    ABORT = "abort"  # Too many failures, giving up


@dataclass
class SchedulerConfig:
    """Tuning knobs for the scheduler.

    All fields have reasonable defaults so users only override what they need.
    """

    # Maximum number of tasks to execute in one run (safety limit)
    max_tasks_per_run: int = 50

    # How many consecutive failures before the scheduler gives up
    max_consecutive_failures: int = 5

    # Whether to persist state after each task
    auto_persist: bool = True

    # Whether to continue after a non-critical failure (skip the task)
    continue_on_failure: bool = True

    # Timeout for a single task execution in seconds
    task_timeout_ms: float = 300_000.0  # 5 min

    # Max tool-call rounds per orchestrated task (prevents infinite loops)
    max_rounds_per_task: int = 20

    # Optional tool-call callback for UI observability
    on_tool_callback: Callable[[str, dict], None] | None = None

    # Parallel execution: run independent tasks concurrently
    parallel: bool = False
    max_parallel: int = 4

    # Goal string for context injection
    goal: str = ""


class Scheduler:
    """Dependency-aware task scheduler.

    Usage:
        scheduler = Scheduler(graph, executor, recovery, verifier, memory_injector, olog)
        scheduler.set_agent_chat(agent.chat)
        decision = await scheduler.run()
    """

    def __init__(
        self,
        graph: TaskGraph,
        executor: Executor,
        recovery: RecoveryManager | None = None,
        verifier: BaseVerifier | None = None,
        memory_injector: MemoryInjector | None = None,
        olog: OrchestrationLogger | None = None,
        config: SchedulerConfig | None = None,
    ):
        self.graph = graph
        self.executor = executor
        self.recovery = recovery or RecoveryManager()
        self.verifier = verifier or NoOpVerifier()
        self.memory_injector = memory_injector or MemoryInjector()
        self.olog = olog or OrchestrationLogger()
        self.config = config or SchedulerConfig()

        # Runtime state
        self._tasks_run: int = 0
        self._failures: list[dict[str, Any]] = []
        self._completed_tasks: list[str] = []
        self._persist_callback: callable | None = None
        self._progress_callback: callable | None = None

    def on_persist(self, callback) -> None:
        """Register a callback for state persistence.  Called after each task."""
        self._persist_callback = callback

    def on_progress(self, callback) -> None:
        """Register a callback(task_node, event) for progress updates.
        Called on every task status transition so the UI can refresh."""
        self._progress_callback = callback

    def _notify_progress(self, task: TaskNode, event: str) -> None:
        """Notify the progress callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(task, event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # main execution loop
    # ------------------------------------------------------------------

    async def run(self) -> SchedulingDecision:
        """Execute tasks until the graph is complete or blocked.

        Returns the final SchedulingDecision so the orchestrator knows
        whether to replan, report success, or abort.
        """
        self.olog.start_run()
        self.olog.emit(EventType.SCHEDULE_PICK, message="Scheduler starting")

        # Wire tool-call callbacks to the executor for observability
        if self.config.on_tool_callback:
            self.executor.set_callbacks(on_tool=self.config.on_tool_callback)

        while self._tasks_run < self.config.max_tasks_per_run:
            # 1. Find ready tasks
            ready = self.graph.get_ready_tasks()

            if not ready:
                # Check if we're done or stuck
                if self.graph.is_complete():
                    self.olog.emit(EventType.SCHEDULE_COMPLETE,
                                   message="All tasks complete",
                                   success=self.graph.is_successful())
                    return (
                        SchedulingDecision.COMPLETE
                        if self.graph.is_successful()
                        else SchedulingDecision.FAILED
                    )

                # Check for blocked tasks — if tasks exist but none are ready,
                # and there are no running tasks, we're stuck
                blocked = self.graph.get_blocked_tasks()
                if blocked and not self._has_running():
                    self.olog.emit(EventType.SCHEDULE_BLOCKED,
                                   message=f"Deadlock: {len(blocked)} tasks blocked",
                                   blocked_count=len(blocked))
                    # Check if any blocked tasks are blocked by FAILED tasks
                    # If so, this is a failure propagation, not a deadlock
                    failed_tasks = self.graph.get_failed_tasks()
                    if failed_tasks:
                        if self.config.continue_on_failure:
                            return SchedulingDecision.FAILED
                        return SchedulingDecision.REPLAN
                    return SchedulingDecision.ABORT

                # Nothing ready but graph not complete — wait (for parallel mode)
                # In sequential mode, this shouldn't happen
                break

            # 2. Pick task(s) to execute
            if self.config.parallel and len(ready) > 1:
                # Parallel mode: execute all independent ready tasks concurrently.
                # Each task gets its own Agent via the factory, so conversation
                # state is isolated — no message interleaving.
                batch = ready[:self.config.max_parallel]
                for t in batch:
                    self.olog.emit(EventType.SCHEDULE_PICK,
                                   task_id=t.id,
                                   task_title=t.title,
                                   priority=t.priority)

                decisions = await asyncio.gather(
                    *[self._execute_single(t) for t in batch],
                    return_exceptions=True,
                )

                for t, decision in zip(batch, decisions):
                    if isinstance(decision, Exception):
                        self.olog.emit(EventType.EXECUTION_ERROR,
                                       task_id=t.id,
                                       error=str(decision))
                        continue
                    if decision == SchedulingDecision.ABORT:
                        return decision
                    if decision == SchedulingDecision.REPLAN:
                        return decision
                    self._tasks_run += 1
                    self._completed_tasks.append(t.id)
            else:
                # Sequential mode: pick the highest-priority ready task.
                task = ready[0]
                self.olog.emit(EventType.SCHEDULE_PICK,
                               task_id=task.id,
                               task_title=task.title,
                               priority=task.priority)

                decision = await self._execute_single(task)

                if decision == SchedulingDecision.ABORT:
                    return decision
                if decision == SchedulingDecision.REPLAN:
                    return decision

                self._tasks_run += 1
                self._completed_tasks.append(task.id)

            # 5. Persist state
            if self.config.auto_persist and self._persist_callback:
                self._persist_callback()

        # Reached task limit
        if self.graph.is_complete():
            return (
                SchedulingDecision.COMPLETE
                if self.graph.is_successful()
                else SchedulingDecision.FAILED
            )
        return SchedulingDecision.ABORT

    # ------------------------------------------------------------------
    # single task execution
    # ------------------------------------------------------------------

    async def _execute_single(self, task: TaskNode) -> SchedulingDecision:
        """Execute one task node through the full lifecycle.

        Lifecycle:
        1. Transition PENDING → RUNNING
        2. Build working memory
        3. Execute via agent
        4. Verify result
        5. Handle success or failure
        """
        task_id = task.id
        task_title = task.title

        # Transition to RUNNING
        old_status = task.status.value
        task.transition_to(TaskStatus.RUNNING)
        self._notify_progress(task, "running")
        self.olog.node_transition(task_id, task_title, old_status, "running",
                                  reason="scheduler picked task")

        # Build working memory
        memory = self.memory_injector.build(
            task_id=task.id,
            graph=self.graph,
            goal=self.config.goal,
            run_id=str(id(self.olog)),
        )

        # Execute
        self.olog.emit(EventType.EXECUTION_START, task_id=task_id, task_title=task_title)
        ctx = TaskContext(task=task, memory=memory)
        result = await self.executor.execute(ctx)
        self.olog.emit(EventType.EXECUTION_END,
                       task_id=task_id,
                       task_title=task_title,
                       success=result.success,
                       duration_ms=result.duration_ms)
        self.olog.execution_timing(task_id, result.duration_ms, result.tokens_used)

        # Verify
        self.olog.emit(EventType.VERIFY_START, task_id=task_id)
        verification = self.verifier.verify(
            result,
            task_metadata=task.metadata,
            working_dir=".",
        )
        task.verification = verification

        if verification.passed:
            self.olog.emit(EventType.VERIFY_PASS, task_id=task_id)
        else:
            self.olog.emit(EventType.VERIFY_FAIL,
                           task_id=task_id,
                           failures=verification.failures)

        # Handle result
        if result.success and verification.passed:
            return self._handle_success(task, result)
        else:
            return self._handle_failure(
                task,
                result.error or "Verification failed",
                verification,
            )

    def _handle_success(
        self, task: TaskNode, result: ExecutionResult
    ) -> SchedulingDecision:
        """Process a successful task execution."""
        task.record_result(result)
        # record_result sets status to SUCCESS, but verification result
        # was already recorded — no override needed
        self.graph.mark_completed(task.id, result)

        # Carry artifacts forward for downstream tasks
        task.artifacts.update(result.artifacts)

        self.recovery.reset_consecutive_failures()
        self._notify_progress(task, "success")
        self.olog.node_transition(
            task.id, task.title, "running", "success",
            reason=f"Completed in {result.duration_ms:.0f}ms",
        )

        # Unblock tasks that were waiting on this one
        self.graph.unblock_dependents(task.id)

        return SchedulingDecision.CONTINUE

    def _handle_failure(
        self,
        task: TaskNode,
        error: str,
        verification,
    ) -> SchedulingDecision:
        """Process a failed task execution — retry, skip, or escalate."""
        self.olog.node_transition(
            task.id, task.title, "running", "failed",
            reason=error[:120],
        )

        # Ask recovery manager what to do
        action = self.recovery.decide(task, error, verification)
        self._failures.append({
            "task_id": task.id,
            "task_title": task.title,
            "error": error,
            "action": action.action,
            "verification_failures": verification.failures if verification else [],
        })

        if action.action == "retry":
            self.olog.retry_event(
                task.id, task.title,
                task.retry_count + 1,
                task.retry_policy.max_retries,
                error,
            )
            # Run rollback hooks before retrying
            self.recovery.run_rollbacks(task)
            # Wait for backoff
            asyncio.create_task(self.recovery.wait_backoff(action.backoff_ms))
            # Prepare for retry
            self.recovery.prepare_retry(task)
            self._notify_progress(task, "retry")
            self.olog.node_transition(
                task.id, task.title, "failed", "pending",
                reason=f"Retry {task.retry_count}/{task.retry_policy.max_retries}",
            )
            # Don't block dependents on retry — task is still PENDING
            return SchedulingDecision.CONTINUE

        elif action.action == "skip":
            task.transition_to(TaskStatus.SKIPPED)
            self.graph.mark_failed(task.id, error)
            self._notify_progress(task, "skipped")
            self.olog.node_transition(
                task.id, task.title, "failed", "skipped",
                reason=action.reason,
            )
            if self.config.continue_on_failure:
                return SchedulingDecision.CONTINUE
            return SchedulingDecision.FAILED

        elif action.action == "replan":
            self.graph.mark_failed(task.id, error)
            self.olog.emit(EventType.REPLAN_TRIGGERED,
                           task_id=task.id,
                           reason=action.reason)
            return SchedulingDecision.REPLAN

        else:  # abort
            self.graph.mark_failed(task.id, error)
            self.olog.node_transition(
                task.id, task.title, "failed", "failed",
                reason=action.reason,
            )
            return SchedulingDecision.ABORT

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _has_running(self) -> bool:
        """Check if any task is currently RUNNING (relevant for parallel mode)."""
        return any(
            TaskGraph._status_eq(n.status, TaskStatus.RUNNING) for n in self.graph.nodes.values()
        )

    def run_summary(self) -> dict[str, Any]:
        """Return a summary of the scheduler run for diagnostics."""
        success, failed, running, pending = self.graph.progress()
        return {
            "tasks_run": self._tasks_run,
            "tasks_completed": len(self._completed_tasks),
            "graph_success": success,
            "graph_failed": failed,
            "graph_running": running,
            "graph_pending": pending,
            "failure_count": len(self._failures),
            "failures": self._failures[-10:],  # Last 10 failures
            "observability": self.olog.summary(),
        }
