"""Core agent loop — state-centric runtime orchestration.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.

Architecture (state-centric, NOT chat-history centric):

    SessionState
      ├── persistent_history   — real conversation only
      ├── repo_summary         — stable repository cognition
      ├── active_files/symbols — current task scope
      ├── working memory       — completed steps, decisions (compactable)
      └── execution state      — bounds, constraints, stop conditions

    build_runtime_messages(state, system_prompt)
      → [system] + [assistant(mem)] + [assistant(repo)] + [assistant(run)]
        + ... persistent_history ...

Ephemeral context (repo, memory, constraints, execution policies) is
rebuilt fresh each turn and prepended to the message list.  It is NEVER
written into persistent_history.  Only real conversation (user messages,
assistant replies, tool calls, tool results) lives there.

This separation means:
- Compression only touches real conversation, not injected metadata.
- Runtime context can be refreshed without polluting history.
- Checkpoint/undo operates on conversation boundaries, not context blobs.
- The LLM sees a clean layered structure with clear attention hierarchy.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import TYPE_CHECKING

from .tools import ALL_TOOLS, get_tool
from .tools.base import Tool
from .tools.agent import AgentTool
from .tools.repo_info import RepoInfoTool
from .prompt import system_prompt
from .runtime.state import SessionState
from .runtime.assembler import (
    build_runtime_messages,
    estimate_ephemeral_tokens,
)
from .history.compression import ContextManager
from .repo.shadow import ShadowGit
from .repo.index import RepoIndex

if TYPE_CHECKING:
    from .llm.client import LLM

logger = logging.getLogger("corecoder.agent")


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.working_dir = os.getcwd()

        # State-centric runtime — replaces the old self.messages
        self.state = SessionState()
        # Backward-compat: raw context_message string (CLI one-shot/REPL mode).
        # When set, the assembler injects it as part of the ephemeral prefix.
        self._ephemeral_context: str | None = None

        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds

        # repository index — structured codebase memory (~/.corecoder/)
        self.repo_index = RepoIndex(os.getcwd())
        if self.repo_index.needs_rebuild() or not self.repo_index.load():
            self.repo_index.build()
            self.repo_index.save_stamp()
        else:
            self.repo_index.load()

        self._system = system_prompt(self.tools)

        # checkpoint: (persistent_history_length, description, git_commit_hash)
        self._checkpoints: list[tuple[int, str, str | None]] = []

        # shadow git for file-level checkpoint / undo / diff
        self.shadow = ShadowGit(os.getcwd())
        self.shadow.init()
        self.shadow.tag_session_start()

        # wire up tools that need Agent references
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
            if isinstance(t, RepoInfoTool):
                t._repo_index = self.repo_index

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_input: str,
        context_message: str | None = None,
        state_updates: dict | None = None,
        on_token=None,
        on_tool=None,
    ) -> str:
        """Process one user message. May involve multiple LLM/tool rounds.

        Supports two modes for runtime context injection:

        1. **Structured** (orchestrated mode): pass ``state_updates`` dict.
           Fields are merged into SessionState.  The assembler rebuilds
           ephemeral prefixes from state fields each turn.

        2. **String** (CLI backward compat): pass ``context_message`` str.
           Stored as ``_ephemeral_context`` and injected as-is into the
           assembler's context layer.

        In both cases, runtime context is EPHEMERAL — it is never written
        into persistent_history.

        Args:
            user_input: The user message (goal + current task).
            context_message: Optional raw context string (CLI backward compat).
            state_updates: Optional dict of SessionState field updates (orchestrated mode).
            on_token: Optional callback for each streamed token.
            on_tool: Optional callback for each tool call.
        """
        # Apply structured state updates (orchestrated mode).
        # These populate SessionState fields so the assembler can build
        # layered ephemeral prefixes.
        if state_updates:
            self.state.apply_state_updates(state_updates)

        # Store raw context string for backward compat (CLI mode).
        # The assembler will inject this as an ephemeral assistant message.
        if context_message:
            self._ephemeral_context = context_message

        # git snapshot before this turn (lightweight: only commits if dirty)
        self.shadow.snapshot(f"checkpoint: {user_input[:60]}")
        head = self._shadow_head()
        self._checkpoints.append(
            (len(self.state.persistent_history), f"user: {user_input[:60]}", head)
        )

        # User message goes into persistent_history — this IS real conversation.
        self.state.persistent_history.append({"role": "user", "content": user_input})

        # Compression only touches persistent_history.  Ephemeral overhead
        # is accounted for so thresholds are accurate.
        ephemeral_overhead = estimate_ephemeral_tokens(self.state, self._system)
        if self._ephemeral_context:
            ephemeral_overhead += len(self._ephemeral_context) // 3
        await self.context.maybe_compress(
            self.state.persistent_history, self.llm,
            ephemeral_overhead=ephemeral_overhead,
        )

        result_text: str | None = None

        for _ in range(self.max_rounds):
            if self.tools:
                text = await self._execute_turn_sse(on_token, on_tool)
                if text is not None:
                    result_text = text
                    break
            else:
                resp = await self.llm.chat(
                    messages=self._build_messages_for_llm(),
                    tools=None,
                    on_token=on_token,
                )
                if not resp.tool_calls:
                    self.state.persistent_history.append(resp.message)
                    result_text = resp.content
                    break

                self.state.persistent_history.append(resp.message)
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = await self._call_tool(tc)
                    self.state.persistent_history.append({
                        "role": "tool", "tool_call_id": tc.id, "content": result,
                    })
                else:
                    results = await self._call_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.state.persistent_history.append({
                            "role": "tool", "tool_call_id": tc.id, "content": result,
                        })

            # Recompute ephemeral overhead after each turn — tool results
            # may have changed what's in active context.
            ephemeral_overhead = estimate_ephemeral_tokens(self.state, self._system)
            if self._ephemeral_context:
                ephemeral_overhead += len(self._ephemeral_context) // 3
            await self.context.maybe_compress(
                self.state.persistent_history, self.llm,
                ephemeral_overhead=ephemeral_overhead,
            )

        # Rebuild repo index after every user turn — files may have changed
        try:
            self.repo_index.build()
        except Exception:
            pass

        return result_text or "(reached maximum tool-call rounds)"

    def reset(self):
        """Clear conversation history and checkpoints.  Fresh SessionState."""
        self.state = SessionState()
        self._ephemeral_context = None
        self._checkpoints.clear()

    # ------------------------------------------------------------------
    # checkpoint / undo
    # ------------------------------------------------------------------

    def undo(self) -> str | None:
        """Restore files and conversation to before the last user turn.

        Operates on persistent_history only — ephemeral context is
        not checkpointed (it's rebuilt from SessionState fields).
        """
        if not self._checkpoints:
            return None

        n_files = len(self.changed_files)

        # If nothing changed at all, don't pretend we undid something
        if n_files == 0 and len(self._checkpoints) == 1:
            return None

        _, desc, commit = self._checkpoints.pop()

        if not self._checkpoints:
            self.state.persistent_history.clear()
            if commit:
                try:
                    self.shadow._git("reset", "--hard", commit)
                except Exception as e:
                    logger.warning("Shadow reset failed: %s", e)
            file_info = f" — restored {n_files} file(s)" if n_files else ""
            return f"{desc}{file_info}"

        # git reset to the pre-turn commit
        if commit:
            try:
                self.shadow._git("reset", "--hard", commit)
            except Exception as e:
                logger.warning("Shadow reset failed: %s", e)

        target_len, _, _ = self._checkpoints[-1]
        self.state.persistent_history = self.state.persistent_history[:target_len]

        suffix = f" — restored {n_files} file(s)" if n_files else ""
        return desc + suffix

    # ------------------------------------------------------------------
    # diff helpers (used by CLI /diff)
    # ------------------------------------------------------------------

    @property
    def changed_files(self) -> list[str]:
        return self.shadow.changed_files()

    @property
    def last_diff(self) -> str:
        return self.shadow.last_diff()

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------

    async def _call_tool(self, tc) -> str:
        tool = get_tool(tc.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tc.name)
            return f"Error: unknown tool '{tc.name}'"
        return await self._invoke(tool, tc.arguments)

    async def _call_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        async def run_one(tc):
            tool = get_tool(tc.name)
            if tool is None:
                return f"Error: unknown tool '{tc.name}'"
            return await self._invoke(tool, tc.arguments)

        return await asyncio.gather(*[run_one(tc) for tc in tool_calls])

    @staticmethod
    async def _invoke(tool: Tool, kwargs: dict) -> str:
        try:
            if inspect.iscoroutinefunction(tool.execute):
                return await tool.execute(**kwargs)
            else:
                return await asyncio.to_thread(tool.execute, **kwargs)
        except TypeError as e:
            logger.warning("Bad arguments for %s: %s", tool.name, e)
            return f"Error: bad arguments for {tool.name}: {e}"
        except Exception as e:
            logger.warning("Error executing %s: %s", tool.name, e)
            return f"Error executing {tool.name}: {e}"

    # ------------------------------------------------------------------
    # SSE execution — start tools immediately as they arrive
    # ------------------------------------------------------------------

    async def _execute_turn_sse(self, on_token=None, on_tool=None) -> str | None:
        """Execute one ReAct turn using SSE streaming.

        Opens an SSE stream from the LLM.  As soon as each tool call's
        arguments form valid JSON, execution starts via asyncio.create_task
        while the stream continues.  This overlaps tool execution with
        LLM response streaming.

        Returns:
            Text content if the LLM responded without tool calls.
            None if tool calls were executed (results appended to persistent_history).
        """
        from .llm.types import LLMResponse, ToolCall as LlmToolCall

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[LlmToolCall] = []
        # (tool_call, future) — started immediately when tool_call event arrives
        pending: list[tuple[LlmToolCall, asyncio.Task]] = []
        prompt_tok = 0
        completion_tok = 0

        async for event in self.llm.chat_sse(
            messages=self._build_messages_for_llm(),
            tools=self._tool_schemas(),
        ):
            if event.type == "text":
                content_parts.append(event.token or "")
                if on_token:
                    ret = on_token(event.token)
                    if inspect.isawaitable(ret):
                        await ret

            elif event.type == "reasoning":
                reasoning_parts.append(event.token or "")
                # Forward reasoning tokens to on_token so the UI shows
                # activity during long thinking phases (DeepSeek-R1 can
                # reason silently for minutes).  Prefix with [think] so
                # the UI can distinguish reasoning from output.
                if on_token:
                    ret = on_token("[think] " + (event.token or ""))
                    if inspect.isawaitable(ret):
                        await ret

            elif event.type == "tool_call" and event.tool_call:
                tc = event.tool_call
                tool_calls.append(tc)
                if on_tool:
                    on_tool(tc.name, tc.arguments)
                # Fire-and-continue: tool executes in background while
                # the LLM stream produces more events
                task = asyncio.create_task(self._call_tool(tc))
                pending.append((tc, task))

            elif event.type == "done":
                if event.usage:
                    prompt_tok = event.usage.get("prompt_tokens", 0)
                    completion_tok = event.usage.get("completion_tokens", 0)

            elif event.type == "error":
                logger.warning("SSE error: %s", event.error)

        # No tool calls → LLM responded with text only
        if not tool_calls:
            resp = LLMResponse(
                content="".join(content_parts),
                reasoning_content="".join(reasoning_parts),
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
            )
            self.state.persistent_history.append(resp.message)
            return resp.content

        # Build assistant message with all tool calls
        resp = LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )
        self.state.persistent_history.append(resp.message)

        # Collect results from background tool executions
        for tc, task in pending:
            try:
                result = await task
            except Exception as e:
                logger.warning("Tool %s failed: %s", tc.name, e)
                result = f"Error executing {tc.name}: {e}"
            self.state.persistent_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        return None  # Signal: tool calls were executed

    # ------------------------------------------------------------------
    # message assembly (replaces old _full_messages)
    # ------------------------------------------------------------------

    def _build_messages_for_llm(self) -> list[dict]:
        """Build the message list for the next LLM inference call.

        Uses the runtime assembler to layer ephemeral context prefixes
        over persistent_history.  The assembler rebuilds the prefix from
        SessionState fields on every call, so state changes take effect
        immediately without polluting conversation history.

        Backward compat: if _ephemeral_context is set (CLI mode), it is
        injected as an additional assistant message before persistent_history.
        """
        messages = build_runtime_messages(self.state, self._system)

        # Inject backward-compat raw context string (CLI one-shot/REPL mode)
        # between the ephemeral prefix and persistent_history.
        # The assembler produces: [system, *ephemeral_prefix, *persistent_history]
        # We need: [system, *ephemeral_prefix, _ephemeral_context, *persistent_history]
        if self._ephemeral_context:
            # Find the split point: everything before persistent_history
            # is ephemeral prefix.  Insert context before history.
            hist_len = len(self.state.persistent_history)
            if hist_len > 0:
                insert_at = len(messages) - hist_len
                messages.insert(insert_at, {
                    "role": "assistant", "content": self._ephemeral_context,
                })
            else:
                messages.append({
                    "role": "assistant", "content": self._ephemeral_context,
                })

        return messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _shadow_head(self) -> str | None:
        """Return the current shadow repo HEAD commit hash, or None."""
        try:
            return self.shadow._git("rev-parse", "HEAD")
        except Exception:
            return None
