"""DAG-based task orchestration layer for CoreCoder.

Sits above the ReAct agent loop, providing structured planning,
dependency-aware scheduling, failure recovery, and context orchestration.

Package structure:
    dag/       — Task graph, models, working memory, recovery
    engine/    — Scheduler, planner, executor, verifier
    context/   — Context Orchestrator (dynamic context assembly)
"""

from corecoder.orchestration.dag import (
    TaskStatus,
    TaskNode,
    ExecutionResult,
    VerificationResult,
    RetryPolicy,
    TaskGraph,
    CycleDetectedError,
    WorkingMemory,
    MemoryInjector,
    RecoveryManager,
    RecoveryAction,
    DefaultRetryPolicy,
    resume_graph_state,
)
from corecoder.orchestration.engine import (
    Scheduler,
    SchedulerConfig,
    SchedulingDecision,
    BasePlanner,
    StaticPlanner,
    LLMPlanner,
    PlanResult,
    Executor,
    TaskContext,
    MaxRoundsExceededError,
    BaseVerifier,
    NoOpVerifier,
    CompositeVerifier,
    TestVerifier,
    LintVerifier,
    OutputVerifier,
    FileExistsVerifier,
)
from corecoder.orchestration.storage import BaseStorage, JSONStorage
from corecoder.orchestration.observability import (
    OrchestrationLogger,
    EventType,
    TaskTransition,
)
from corecoder.orchestration.orchestrator import Orchestrator, OrchestratorConfig

__all__ = [
    # DAG
    "TaskStatus", "TaskNode", "ExecutionResult", "VerificationResult",
    "RetryPolicy", "TaskGraph", "CycleDetectedError",
    "WorkingMemory", "MemoryInjector",
    "RecoveryManager", "RecoveryAction", "DefaultRetryPolicy", "resume_graph_state",
    # Engine
    "Scheduler", "SchedulerConfig", "SchedulingDecision",
    "BasePlanner", "StaticPlanner", "LLMPlanner", "PlanResult",
    "Executor", "TaskContext", "MaxRoundsExceededError",
    "BaseVerifier", "NoOpVerifier", "CompositeVerifier",
    "TestVerifier", "LintVerifier", "OutputVerifier", "FileExistsVerifier",
    # Infrastructure
    "BaseStorage", "JSONStorage",
    "OrchestrationLogger", "EventType", "TaskTransition",
    "Orchestrator", "OrchestratorConfig",
]
