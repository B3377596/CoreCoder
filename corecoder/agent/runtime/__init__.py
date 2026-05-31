"""Runtime state management  state-centric orchestration.

SessionState is the single source of truth for the agent's runtime cognition.
It cleanly separates:

- persistent_history: real conversation (user, assistant replies, tool traces)
- ephemeral context: repo cognition, working memory, execution policies  rebuilt
  each turn, never written into conversation history
"""

from corecoder.agent.runtime.state import SessionState
from corecoder.agent.runtime.assembler import build_runtime_messages, estimate_ephemeral_tokens
from corecoder.agent.runtime.staged import (
    AgentRuntime,
    ExecutionResult,
    GlobalStateManager,
    GlobalTaskState,
    LocalReactExecutor,
    RuntimeEvaluation,
    StageEvaluator,
    StageExecutor,
    StagePlan,
    ThinkDecision,
    ThinkEngine,
)
from corecoder.agent.runtime.verification import PatchAnalysis, VerificationResult, VerificationPolicyEngine

__all__ = [
    "SessionState",
    "build_runtime_messages",
    "estimate_ephemeral_tokens",
    "StagePlan",
    "ExecutionResult",
    "GlobalTaskState",
    "ThinkDecision",
    "RuntimeEvaluation",
    "ThinkEngine",
    "LocalReactExecutor",
    "StageExecutor",
    "StageEvaluator",
    "GlobalStateManager",
    "AgentRuntime",
    "PatchAnalysis",
    "VerificationResult",
    "VerificationPolicyEngine",
]
