"""Dynamic runtime message assembly.

Replaces the old _full_messages() which blindly concatenated
[system] + self.messages.  Instead, builds a fresh message list
each turn by layering:

    1. system          — stable rules (always)
    2. assistant(mem)  — working memory (ephemeral, rebuilt each turn)
    3. assistant(repo) — repository cognition (ephemeral)
    4. assistant(run)  — execution constraints (ephemeral)
    5. persistent_history — real conversation (append-only)

Layers 2-4 are reconstructed from SessionState on every call.
They are NEVER appended to persistent_history — only the assembler
prepends them before sending to the LLM.

Design principle: the LLM sees a clean, layered message structure
where environment/state/memory are clearly separated from real
conversation.  This gives the model better attention hierarchy
and prevents runtime metadata from polluting conversation compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from corecoder.history.compression import count_tokens

if TYPE_CHECKING:
    from corecoder.runtime.state import SessionState


def build_runtime_messages(
    state: SessionState,
    system_prompt: str,
    current_turn: dict | None = None,
) -> list[dict]:
    """Build the full message list for one LLM inference turn.

    Layers (in order):
    1. system message (always)
    2. assistant(memory) — working memory, if any fields are non-empty
    3. assistant(repo)   — repository context, if any fields are non-empty
    4. assistant(runtime) — execution constraints, if any fields are non-empty
    5. current_turn user message (if provided — used by CLI one-shot path)
    6. persistent_history — the real conversation so far

    Args:
        state: Current SessionState with all runtime fields.
        system_prompt: The stable system prompt string.
        current_turn: Optional user message dict (e.g. {"role": "user", "content": "..."}).
                      When provided, it's inserted before persistent_history.

    Returns:
        Flat list of message dicts ready to send to the LLM.
    """
    messages: list[dict] = []

    # Layer 1: system (always first)
    messages.append({"role": "system", "content": system_prompt})

    # Layer 2: assistant(memory) — working memory
    mem_content = _build_memory_prefix(state)
    if mem_content:
        messages.append({"role": "assistant", "content": mem_content})

    # Layer 3: assistant(repo) — repository cognition
    repo_content = _build_repo_prefix(state)
    if repo_content:
        messages.append({"role": "assistant", "content": repo_content})

    # Layer 4: assistant(runtime) — execution constraints
    runtime_content = _build_runtime_prefix(state)
    if runtime_content:
        messages.append({"role": "assistant", "content": runtime_content})

    # Layer 5: current turn user message (for one-shot CLI mode)
    if current_turn is not None:
        messages.append(current_turn)

    # Layer 6: persistent conversation history
    messages.extend(state.persistent_history)

    return messages


def estimate_ephemeral_tokens(state: SessionState, system_prompt: str) -> int:
    """Estimate token count of the ephemeral prefix layers.

    This is the overhead ABOVE persistent_history.  Used by compression
    to determine when to trigger summarization — compression thresholds
    should account for this overhead.

    Uses ``count_tokens`` from the compression module for consistency.
    """
    overhead = count_tokens(system_prompt)
    mem = _build_memory_prefix(state)
    if mem:
        overhead += count_tokens(mem)
    repo = _build_repo_prefix(state)
    if repo:
        overhead += count_tokens(repo)
    runtime = _build_runtime_prefix(state)
    if runtime:
        overhead += count_tokens(runtime)
    return max(1, overhead)


# ------------------------------------------------------------------
# Layer builders — each returns a string or ""
# ------------------------------------------------------------------


def _build_memory_prefix(state: SessionState) -> str:
    """Build the assistant(memory) prefix — completed work and decisions.

    Goal and current task are deliberately EXCLUDED — they belong in the
    user message, not duplicated here.  This layer only carries "what we've
    done so far" to give the agent continuity without polluting the instruction.
    """
    parts: list[str] = []

    if state.completed_steps:
        steps = "\n".join(f"- {s}" for s in state.completed_steps[-15:])
        parts.append(f"## Completed Steps\n{steps}")

    if state.important_decisions:
        decisions = "\n".join(f"- {d}" for d in state.important_decisions[-10:])
        parts.append(f"## Key Decisions\n{decisions}")

    if not parts:
        return ""
    return "[WORKING MEMORY]\n" + "\n\n".join(parts)


def _build_repo_prefix(state: SessionState) -> str:
    """Build the assistant(repo) prefix — repository structure knowledge."""
    parts: list[str] = []

    if state.repo_summary:
        parts.append(state.repo_summary)

    if state.active_files:
        files = "\n".join(f"- {f}" for f in state.active_files[:15])
        parts.append(f"## Active Files\n{files}")

    if state.active_symbols:
        syms = ", ".join(state.active_symbols[:20])
        parts.append(f"## Active Symbols\n{syms}")

    if not parts:
        return ""
    return "[REPOSITORY CONTEXT]\n" + "\n\n".join(parts)


def _build_runtime_prefix(state: SessionState) -> str:
    """Build the assistant(runtime) prefix — execution boundaries and constraints."""
    parts: list[str] = []

    if state.allowed_actions:
        parts.append(f"**ALLOWED**: {', '.join(state.allowed_actions)}")

    if state.forbidden_actions:
        parts.append(f"**FORBIDDEN**: {', '.join(state.forbidden_actions)}")

    if state.stop_conditions:
        parts.append(f"**STOP WHEN**: {state.stop_conditions}")

    if state.downstream_tasks:
        tasks = ", ".join(state.downstream_tasks)
        parts.append(f"**DOWNSTREAM TASKS (do NOT do these)**: {tasks}")

    if state.constraints:
        constraints = "\n".join(f"- {c}" for c in state.constraints)
        parts.append(f"## Constraints\n{constraints}")

    if state.failures:
        failures = "\n".join(f"- {f}" for f in state.failures[-5:])
        parts.append(f"## Recent Failures (do NOT repeat)\n{failures}")

    if not parts:
        return ""
    return "[EXECUTION CONSTRAINTS]\n" + "\n".join(parts)
