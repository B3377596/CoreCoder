"""Runtime state management  state-centric orchestration.

SessionState is the single source of truth for the agent's runtime cognition.
It cleanly separates:

- persistent_history: real conversation (user, assistant replies, tool traces)
- ephemeral context: repo cognition, working memory, execution policies  rebuilt
  each turn, never written into conversation history
"""

from corecoder.agent.runtime.state import SessionState
from corecoder.agent.runtime.assembler import build_runtime_messages, estimate_ephemeral_tokens

__all__ = ["SessionState", "build_runtime_messages", "estimate_ephemeral_tokens"]
