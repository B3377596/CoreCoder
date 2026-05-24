"""DAG-based task orchestration layer for CoreCoder.

Sits above the ReAct agent loop, providing:
- Structured planning (user goal → task DAG)
- Dependency-aware scheduling
- Failure recovery with partial retries
- Execution state persistence
- Dynamic replanning extension points
- Multi-agent compatibility architecture

Architecture:
    User Goal → Planner → TaskGraph (DAG) → Scheduler → Executor → Verifier
                                                                    ↓
                                                              Replan if needed
"""

from corecoder.orchestration.models import (
    TaskStatus,
    TaskNode,
    ExecutionResult,
    VerificationResult,
    RetryPolicy,
)
from corecoder.orchestration.graph import TaskGraph, CycleDetectedError
from corecoder.orchestration.scheduler import Scheduler, SchedulingDecision
from corecoder.orchestration.planner import (
    BasePlanner,
    StaticPlanner,
    LLMPlanner,
    PlanResult,
)
from corecoder.orchestration.executor import Executor, TaskContext
from corecoder.orchestration.verifier import (
    BaseVerifier,
    NoOpVerifier,
    CompositeVerifier,
    TestVerifier,
    LintVerifier,
)
from corecoder.orchestration.storage import BaseStorage, JSONStorage
from corecoder.orchestration.recovery import RecoveryManager, DefaultRetryPolicy
from corecoder.orchestration.memory import WorkingMemory, MemoryInjector
from corecoder.orchestration.observability import (
    OrchestrationLogger,
    EventType,
    TaskTransition,
)
from corecoder.orchestration.orchestrator import Orchestrator, OrchestratorConfig

__all__ = [
    # Models
    "TaskStatus",
    "TaskNode",
    "ExecutionResult",
    "VerificationResult",
    "RetryPolicy",
    # Graph
    "TaskGraph",
    "CycleDetectedError",
    # Scheduler
    "Scheduler",
    "SchedulingDecision",
    # Planner
    "BasePlanner",
    "StaticPlanner",
    "LLMPlanner",
    "PlanResult",
    # Executor
    "Executor",
    "TaskContext",
    # Verifier
    "BaseVerifier",
    "NoOpVerifier",
    "CompositeVerifier",
    "TestVerifier",
    "LintVerifier",
    # Storage
    "BaseStorage",
    "JSONStorage",
    # Recovery
    "RecoveryManager",
    "DefaultRetryPolicy",
    # Memory
    "WorkingMemory",
    "MemoryInjector",
    # Observability
    "OrchestrationLogger",
    "EventType",
    "TaskTransition",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
]
