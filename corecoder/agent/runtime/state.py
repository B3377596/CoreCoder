"""SessionState ? the single source of truth for agent runtime cognition.

Design invariants:
- persistent_history stores ONLY real conversation: user messages, assistant
  text replies, assistant tool_call messages, and tool result messages.
- Ephemeral context (repo summaries, working memory, constraints, execution
  policies) lives in named fields and is injected as transient message prefixes
  by the assembler ? NEVER appended to persistent_history.
- Working memory fields (completed_steps, important_decisions, failures) are
  compactable ? they can be summarized or truncated without losing conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """State-centric runtime cognition.

    Split into layers with distinct lifecycles:

    persistent_history  ? session-long. Only real user/assistant/tool turns.
    repo_summary        ? session-long, lazy-refreshed.
    active_files/symbols? task-long. Reset per task.
    completed_steps etc.? task-long, compactable.
    allowed_actions etc.? execution-long. Reset per task execution.
    """

    # === Persistent conversation (session-long) ===
    # Only: user messages, assistant real replies, tool_call messages,
    # tool result messages.  NEVER contains injected context.
    persistent_history: list[dict] = field(default_factory=list)

    # === Stable repository cognition (session-long, lazy refresh) ===
    repo_summary: str = ""
    # symbol_graph / dependency_graph are stored externally (Retriever owns them).
    # SessionState caches the textual summary only.

    # === Active task context (task-long, reset per scheduler task) ===
    active_files: list[str] = field(default_factory=list)
    active_symbols: list[str] = field(default_factory=list)
    current_task: str = ""
    current_goal: str = ""

    # === Working memory (task-long, compactable) ===
    completed_steps: list[str] = field(default_factory=list)
    important_decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # === Execution state (execution-long, reset per task) ===
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    stop_conditions: str = ""
    downstream_tasks: list[str] = field(default_factory=list)

    # === Retrieval cache (session-long, invalidated on repo change) ===
    retrieval_cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # state mutation
    # ------------------------------------------------------------------

    def apply_state_updates(self, updates: dict[str, Any]) -> None:
        """Merge structured state updates from the ContextOrchestrator.

        Only updates fields that are present in the dict.  Fields not
        mentioned are left unchanged.  Lists are REPLACED, not extended,
        since the orchestrator sends the complete current set each time.

        This is the primary integration point: the Executor calls
        orchestrator.build_state_updates() and passes the result here.
        """
        list_fields = (
            "active_files", "active_symbols", "completed_steps",
            "important_decisions", "constraints", "failures",
            "allowed_actions", "forbidden_actions", "downstream_tasks",
        )
        str_fields = (
            "repo_summary", "current_task", "current_goal", "stop_conditions",
        )

        for key in list_fields:
            if key in updates:
                setattr(self, key, list(updates[key]))

        for key in str_fields:
            if key in updates:
                setattr(self, key, str(updates[key]))

        # retrieval_cache is special: merge, don't replace
        if "retrieval_cache" in updates:
            self.retrieval_cache.update(updates["retrieval_cache"])

    def record_memory(self, text: str) -> None:
        """Extract key decisions from agent output and append to working memory.

        Called after each task execution so the next task knows what happened.
        Best-effort extraction ? the real ground truth is in the git diff.
        """
        if not text.strip():
            return
        lines = text.strip().splitlines()
        # Keep first and last few lines as a compact summary
        if len(lines) <= 3:
            self.completed_steps.append(text.strip()[:200])
        else:
            self.completed_steps.append(
                "\n".join(lines[:2]) + "\n..." + "\n" + lines[-1]
            )

    def record_failure(self, error_msg: str) -> None:
        """Record a failure so the agent can avoid repeating it."""
        self.failures.append(error_msg[:300])
        if len(self.failures) > 10:
            self.failures = self.failures[-10:]

    def should_refresh_repo(self, task_description: str) -> bool:
        """Return True if the current task has drifted from cached repo context.

        Heuristic: if the task description mentions files or concepts not
        in our active_files/active_symbols sets, we may need a fresh retrieval.
        Always returns True if repo_summary is empty (first use).
        """
        if not self.repo_summary:
            return True
        if not self.active_files and not self.active_symbols:
            return True
        return False  # For now, trust the cache.  Future: semantic drift detection.

    def compact_working_memory(self, max_steps: int = 20, max_decisions: int = 10) -> bool:
        """Truncate working memory lists when they exceed limits.

        Returns True if anything was truncated.  The caller should consider
        whether a full summarization pass is needed.
        """
        changed = False
        if len(self.completed_steps) > max_steps:
            self.completed_steps = self.completed_steps[-max_steps:]
            changed = True
        if len(self.important_decisions) > max_decisions:
            self.important_decisions = self.important_decisions[-max_decisions:]
            changed = True
        if len(self.failures) > 10:
            self.failures = self.failures[-10:]
            changed = True
        return changed

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for session persistence.

        Only persists fields that survive across sessions:
        - persistent_history (the real conversation)
        - repo_summary (stable, can be rebuilt on load if stale)
        - completed_steps, important_decisions (compact working memory)
        """
        return {
            "persistent_history": self.persistent_history,
            "repo_summary": self.repo_summary,
            "completed_steps": self.completed_steps[-20:],
            "important_decisions": self.important_decisions[-10:],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Restore from a previously serialized dict."""
        return cls(
            persistent_history=data.get("persistent_history", []),
            repo_summary=data.get("repo_summary", ""),
            completed_steps=data.get("completed_steps", []),
            important_decisions=data.get("important_decisions", []),
        )

    def clear_task_state(self) -> None:
        """Reset task-level fields without touching persistent history.

        Called when moving to a new task in orchestrated mode, or when
        the user starts a fresh request in REPL mode.
        """
        self.current_task = ""
        self.current_goal = ""
        self.completed_steps.clear()
        self.important_decisions.clear()
        self.constraints.clear()
        self.failures.clear()
        self.allowed_actions.clear()
        self.forbidden_actions.clear()
        self.stop_conditions = ""
        self.downstream_tasks.clear()
        self.active_files.clear()
        self.active_symbols.clear()
