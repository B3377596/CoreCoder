"""Top-level orchestrator ?*the single entry point for DAG-based execution.

The Orchestrator owns the full lifecycle:
    User Goal ?*Plan ?*Schedule ?*Execute ?*Verify ?*(Replan) ?*Done

It wires together Planner, TaskGraph, Scheduler, Executor, Verifier,
RecoveryManager, MemoryInjector, Storage, and Observability into a
single runnable pipeline.

Usage:
    orchestrator = Orchestrator(config)
    orchestrator.set_agent(agent)
    result = await orchestrator.run("Build a REST API for todos")

For replanning:
    orchestrator.register_replan_hook(my_replan_function)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from corecoder.agent.dag.models import TaskNode, TaskStatus
from corecoder.agent.dag.graph import TaskGraph
from corecoder.agent.workflow.planner import (
    BasePlanner,
    StaticPlanner,
    LLMPlanner,
    PlanResult,
)
from corecoder.agent.workflow.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingDecision,
)
from corecoder.agent.workflow.executor import Executor
from corecoder.agent.workflow.verifier import BaseVerifier
from corecoder.agent.dag.recovery import RecoveryManager
from corecoder.agent.dag.memory import MemoryInjector
from corecoder.infra.storage import BaseStorage, JSONStorage
from corecoder.infra.observability import OrchestrationLogger, EventType


@dataclass
class OrchestratorConfig:
    """Top-level configuration for the orchestration pipeline.

    Orchestrator-unique settings live here.  Scheduler-specific settings
    are in the embedded ``scheduler`` SchedulerConfig ?*no field duplication.
    """

    goal: str = ""
    working_dir: str = "."
    max_replans: int = 3
    auto_persist: bool = True
    storage_dir: str = ".corecoder/orchestration"

    # Planner configuration
    planner_type: str = "static"  # "static", "llm"

    # Verifier configuration
    run_tests: bool = False
    test_command: str = ""
    run_lint: bool = False
    lint_command: str = ""

    # Scheduler configuration ?*single source of truth for scheduling knobs
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def to_scheduler_config(self, goal: str = "") -> SchedulerConfig:
        """Produce the final SchedulerConfig with goal merged in."""
        sc = SchedulerConfig(
            max_tasks_per_run=self.scheduler.max_tasks_per_run,
            max_consecutive_failures=self.scheduler.max_consecutive_failures,
            auto_persist=self.auto_persist,
            continue_on_failure=self.scheduler.continue_on_failure,
            task_timeout_ms=self.scheduler.task_timeout_ms,
            max_rounds_per_task=self.scheduler.max_rounds_per_task,
            on_tool_callback=self.scheduler.on_tool_callback,
            on_token_callback=self.scheduler.on_token_callback,
            parallel=self.scheduler.parallel,
            max_parallel=self.scheduler.max_parallel,
            goal=goal,
        )
        return sc


@dataclass
class OrchestratorResult:
    """Final output of an orchestration run."""

    success: bool
    goal: str = ""
    plan_summary: str = ""
    tasks_total: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    replans_used: int = 0
    total_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    run_id: str = ""
    graph: TaskGraph | None = None
    summary: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Top-level coordinator for DAG-based task execution.

    Owns the lifecycle: Plan ?*Schedule ?*Execute ?*Verify ?*(Replan).

    The orchestrator is deliberately stateless between runs.  Each call
    to run() creates a fresh pipeline.  This makes it safe to reuse
    across multiple goals without cross-contamination.
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()

        # Components ?*created per-run or lazily
        self._planner: BasePlanner | None = None
        self._storage: BaseStorage | None = None
        self._agent_chat_fn: Callable[..., Awaitable[str]] | None = None
        self._agent_instance: Any = None  # full Agent object for cloning in parallel mode
        self._context_orchestrator: Any = None  # ContextOrchestrator for dynamic context

        # Replan hooks
        self._replan_hooks: list[Callable[[PlanResult, dict[str, Any]], PlanResult]] = []

        # Progress callback ?*notified on every task transition
        self._progress_callback: Callable[[TaskNode, str], None] | None = None

        # Last run state (for introspection)
        self.last_graph: TaskGraph | None = None
        self.last_result: OrchestratorResult | None = None

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_agent(self, agent_chat_fn: Callable[..., Awaitable[str]], agent_instance: Any = None) -> None:
        """Inject the agent's chat method for task execution.

        Optionally pass the full Agent instance for parallel mode,
        where each task needs a fresh Agent clone.
        """
        self._agent_chat_fn = agent_chat_fn
        if agent_instance is not None:
            self._agent_instance = agent_instance

    def set_planner(self, planner: BasePlanner) -> None:
        self._planner = planner

    def set_storage(self, storage: BaseStorage) -> None:
        self._storage = storage

    def set_context_orchestrator(self, orchestrator: Any) -> None:
        """Inject a ContextOrchestrator for dynamic context assembly.

        When set, each task execution uses the orchestrator's pipeline
        (collect ?*rank ?*deduplicate ?*compress ?*budget) instead of
        the flat prompt builder in the Executor.
        """
        self._context_orchestrator = orchestrator

    def on_progress(self, callback: Callable[[TaskNode, str], None]) -> None:
        """Register a callback(task_node, event) for progress updates.

        Events: "running", "success", "failed", "retry", "skipped"
        Called on every task status transition so the UI can refresh.
        """
        self._progress_callback = callback

    def register_replan_hook(
        self, hook: Callable[[PlanResult, dict[str, Any]], PlanResult]
    ) -> None:
        """Register a function that modifies plans during replanning.

        Hooks are called in registration order when the scheduler
        signals that replanning is needed.  Each hook receives the
        current plan and the failure context, and returns a modified plan.
        """
        self._replan_hooks.append(hook)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        goal: str | None = None,
        context: dict[str, Any] | None = None,
        plan: PlanResult | None = None,
    ) -> OrchestratorResult:
        """Execute a user goal through the full orchestration pipeline.

        Args:
            goal: The user's high-level objective.
            context: Additional context (repo info, constraints, etc.)
            plan: Pre-made PlanResult.  If provided, skips the planning phase.
                  Use this when you've already called planner.aplan() to show
                  the graph before execution.

        Returns:
            OrchestratorResult with execution summary.
        """
        goal = goal or self.config.goal
        if not goal:
            raise ValueError("No goal specified")

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        olog = OrchestrationLogger(name=run_id)
        olog.emit(EventType.GRAPH_CREATED, message=f"Starting run for: {goal}")

        start_time = asyncio.get_event_loop().time()

        # ---- Phase 1: Plan (skip if pre-made plan provided) ----
        if plan is not None:
            olog.emit(EventType.PLAN_COMPLETE,
                      tasks=plan.graph.node_count,
                      summary=plan.plan_summary)
        else:
            olog.emit(EventType.PLAN_START, goal=goal)
            planner = self._get_planner()
            # Use async planning when available (LLMPlanner requires it)
            if hasattr(planner, 'aplan') and callable(planner.aplan):
                plan = await planner.aplan(goal, context)
            else:
                plan = planner.plan(goal, context)
            olog.emit(EventType.PLAN_COMPLETE,
                      tasks=plan.graph.node_count,
                      summary=plan.plan_summary)

        graph = plan.graph

        # ---- Phase 2: Execute ----
        replans_used = 0
        final_decision = SchedulingDecision.CONTINUE
        
        # Replan while the scheduler explicitly requests it and we still have budget.
        while replans_used <= self.config.max_replans:
            decision = await self._execute_graph(graph, goal, plan, run_id, olog)
            final_decision = decision

            if decision == SchedulingDecision.COMPLETE:
                break
            elif decision == SchedulingDecision.REPLAN:
                replans_used += 1
                if replans_used > self.config.max_replans:
                    break
                olog.emit(EventType.REPLAN_TRIGGERED,
                          attempt=replans_used,
                          max_replans=self.config.max_replans)
                plan = self._do_replan(plan, graph, run_id)
                graph = plan.graph
                olog.emit(EventType.REPLAN_COMPLETE,
                          tasks=graph.node_count)
            else:
                # FAILED or ABORT
                break

        # ---- Phase 3: Report ----
        total_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
        success_count, failed_count, _, pending_count = graph.progress()

        # Save final state
        if self.config.auto_persist:
            self._get_storage().save_graph(graph.to_dict())
            self._get_storage().save_run_log(run_id, {
                "goal": goal,
                "run_id": run_id,
                "success": final_decision == SchedulingDecision.COMPLETE,
                "plan_summary": plan.plan_summary,
                "tasks_total": graph.node_count,
                "tasks_succeeded": success_count,
                "tasks_failed": failed_count,
                "replans_used": replans_used,
                "total_duration_ms": total_ms,
                "graph": graph.to_dict(),
                "observability": olog.summary(),
            })

        result = OrchestratorResult(
            success=final_decision == SchedulingDecision.COMPLETE and failed_count == 0,
            goal=goal,
            plan_summary=plan.plan_summary,
            tasks_total=graph.node_count,
            tasks_succeeded=success_count,
            tasks_failed=failed_count,
            tasks_skipped=pending_count,
            replans_used=replans_used,
            total_duration_ms=total_ms,
            errors=[],
            run_id=run_id,
            graph=graph,
            summary=olog.summary(),
        )

        self.last_graph = graph
        self.last_result = result
        return result

    async def resume(self, run_id: str) -> OrchestratorResult:
        """Resume an interrupted run from persistent storage."""
        storage = self._get_storage()
        log_data = storage.load_run_log(run_id)
        if log_data is None:
            raise ValueError(f"Run not found: {run_id}")

        graph_data = log_data.get("graph")
        if graph_data is None:
            raise ValueError(f"No graph data in run: {run_id}")

        graph = TaskGraph.from_dict(graph_data)
        goal = log_data.get("goal", "")

        # Reset RUNNING tasks to PENDING
        from corecoder.agent.dag.recovery import resume_graph_state
        resume_graph_state(graph, graph_data)

        olog = OrchestrationLogger(name=f"resume_{run_id}")
        olog.emit(EventType.GRAPH_LOADED, run_id=run_id, goal=goal)

        plan = PlanResult(graph=graph, plan_summary=log_data.get("plan_summary", ""))

        decision = await self._execute_graph(graph, goal, plan, run_id, olog)

        success_count, failed_count, _, pending_count = graph.progress()
        return OrchestratorResult(
            success=decision == SchedulingDecision.COMPLETE and failed_count == 0,
            goal=goal,
            plan_summary=plan.plan_summary,
            tasks_total=graph.node_count,
            tasks_succeeded=success_count,
            tasks_failed=failed_count,
            tasks_skipped=pending_count,
            run_id=run_id,
            graph=graph,
        )

    # ------------------------------------------------------------------
    # internal phases
    # ------------------------------------------------------------------

    async def _execute_graph(
        self,
        graph: TaskGraph,
        goal: str,
        plan: PlanResult,
        run_id: str,
        olog: OrchestrationLogger,
    ) -> SchedulingDecision:
        """Build and run the scheduler for a given graph."""
        # Build subcomponents
        scheduler_config = self.config.to_scheduler_config(goal)
        executor = Executor(
            agent_chat_fn=self._agent_chat_fn,
            max_rounds_per_task=scheduler_config.max_rounds_per_task,
        )

        # Wire ContextOrchestrator ?*enables dynamic context assembly by default.
        # If an external orchestrator was injected via set_context_orchestrator(),
        # use that.  Otherwise, create one from the agent instance.
        if self._context_orchestrator is not None:
            executor.set_context_orchestrator(self._context_orchestrator)
        elif self._agent_instance is not None:
            from corecoder.context.orchestrator import ContextOrchestrator
            co = ContextOrchestrator(
                working_dir=".",
                repo_index=(
                    self._agent_instance.repo_index
                    if self._agent_instance is not None
                    else None
                ),
            )
            executor.set_context_orchestrator(co)

        # In parallel mode, each task gets a fresh Agent clone to prevent
        # conversation interleaving.  The factory uses the same LLM client
        # and tools as the main agent but starts with clean messages.
        '''if self.config.parallel and self._agent_instance is not None:
            main = self._agent_instance
            def _agent_factory():
                from corecoder.agent import Agent as AgentCls
                return AgentCls(
                    llm=main.llm,
                    tools=[t for t in main.tools if t.name != "agent"],
                    max_context_tokens=main.context.max_tokens,
                    max_rounds=self.config.max_rounds_per_task,
                )
            executor.set_agent_factory(_agent_factory)'''

        recovery = RecoveryManager(
            max_consecutive_failures=scheduler_config.max_consecutive_failures,
        )
        verifier = self._build_verifier()
        memory_injector = MemoryInjector()

        scheduler = Scheduler(
            graph=graph,
            executor=executor,
            recovery=recovery,
            verifier=verifier,
            memory_injector=memory_injector,
            olog=olog,
            config=scheduler_config,
        )

        # Wire persistence callback
        if self.config.auto_persist:
            storage = self._get_storage()
            scheduler.on_persist(lambda: storage.save_graph(graph.to_dict()))

        # Wire progress callback
        if self._progress_callback:
            scheduler.on_progress(self._progress_callback)

        decision = await scheduler.run()
        return decision

    def _do_replan(
        self,
        plan: PlanResult,
        graph: TaskGraph,
        run_id: str,
    ) -> PlanResult:
        """Run replan hooks and the planner's replan method."""
        # Collect failure context from the graph
        failure_context = {
            "failed_tasks": [
                {"id": n.id, "title": n.title, "error": n.error}
                for n in graph.get_failed_tasks()
            ],
            "blocked_tasks": [
                {"id": n.id, "title": n.title}
                for n in graph.get_blocked_tasks()
            ],
        }

        # Run planner's built-in replan
        if self._planner:
            plan = self._planner.replan(plan, failure_context)

        # Run registered hooks
        for hook in self._replan_hooks:
            try:
                plan = hook(plan, failure_context)
            except Exception:
                pass

        return plan

    def _build_verifier(self) -> BaseVerifier:
        """Build a verifier using the VerificationPolicyEngine.

        The engine implements verify() directly ?*at call time it inspects
        the patch and dynamically selects appropriate verifiers.
        """
        from corecoder.agent.workflow.verifier import VerificationPolicyEngine
        return VerificationPolicyEngine()

    # ------------------------------------------------------------------
    # lazy init helpers
    # ------------------------------------------------------------------

    def _get_planner(self) -> BasePlanner:
        if self._planner is not None:
            return self._planner
        self._planner = StaticPlanner()
        return self._planner

    def _get_storage(self) -> BaseStorage:
        if self._storage is not None:
            return self._storage
        self._storage = JSONStorage(base_dir=self.config.storage_dir)
        return self._storage

