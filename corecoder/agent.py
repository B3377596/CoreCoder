"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.

All I/O is async: LLM streaming, tool execution, and context compression
happen without blocking the event loop.  Multiple tool calls in a single
response run concurrently via asyncio.gather, matching Claude Code's
StreamingToolExecutor pattern.

Sync tools (bash, file I/O) run via ``asyncio.to_thread`` so they never
stall the event loop.  Async tools (sub-agent) run directly on the loop.

File changes are tracked via a shadow git repository (``ShadowGit``) so
checkpoint / undo / diff use real git snapshots rather than ad-hoc file
backups.  The shadow repo lives under ``~/.corecoder/shadow/`` and does
not touch the user's ``.git``.
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
        self.messages: list[dict] = []
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

        # checkpoint: (messages_length, description, git_commit_hash)
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

    async def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds.

        Uses SSE streaming when tools are available: tool calls are detected
        and executed as soon as their arguments form valid JSON, without
        waiting for the full LLM response.
        """
        # git snapshot before this turn (lightweight: only commits if dirty)
        self.shadow.snapshot(f"checkpoint: {user_input[:60]}")
        # record head commit so undo can find the right ref
        head = self._shadow_head()
        self._checkpoints.append((len(self.messages), f"user: {user_input[:60]}", head))

        self.messages.append({"role": "user", "content": user_input})
        await self.context.maybe_compress(self.messages, self.llm)

        result_text: str | None = None

        for _ in range(self.max_rounds):
            if self.tools:
                text = await self._execute_turn_sse(on_token, on_tool)
                if text is not None:
                    result_text = text
                    break
            else:
                resp = await self.llm.chat(
                    messages=self._full_messages(),
                    tools=None,
                    on_token=on_token,
                )
                if not resp.tool_calls:
                    self.messages.append(resp.message)
                    result_text = resp.content
                    break

                self.messages.append(resp.message)
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = await self._call_tool(tc)
                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": result,
                    })
                else:
                    results = await self._call_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool", "tool_call_id": tc.id, "content": result,
                        })

            await self.context.maybe_compress(self.messages, self.llm)

        # Rebuild repo index after every user turn — files may have changed
        try:
            self.repo_index.build()
        except Exception:
            pass

        return result_text or "(reached maximum tool-call rounds)"

    def reset(self):
        """Clear conversation history and checkpoints."""
        self.messages.clear()
        self._checkpoints.clear()

    # ------------------------------------------------------------------
    # checkpoint / undo
    # ------------------------------------------------------------------

    def undo(self) -> str | None:
        """Restore files and conversation to before the last user turn."""
        if not self._checkpoints:
            return None

        _, desc, commit = self._checkpoints.pop()

        # count changed files in working tree before resetting
        n_files = len(self.changed_files)

        if not self._checkpoints:
            self.messages.clear()
            # restore working tree to HEAD (pre-turn snapshot)
            if commit:
                try:
                    self.shadow._git("reset", "--hard", commit)
                except Exception as e:
                    logger.warning("Shadow reset failed: %s", e)
            return f"(returned to initial state){' — restored ' + str(n_files) + ' file(s)' if n_files else ''}"

        # git reset to the pre-turn commit
        if commit:
            try:
                self.shadow._git("reset", "--hard", commit)
            except Exception as e:
                logger.warning("Shadow reset failed: %s", e)

        target_len, _, _ = self._checkpoints[-1]
        self.messages = self.messages[:target_len]

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
            None if tool calls were executed (results appended to messages).
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
            messages=self._full_messages(),
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
            self.messages.append(resp.message)
            return resp.content

        # Build assistant message with all tool calls
        resp = LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )
        self.messages.append(resp.message)

        # Collect results from background tool executions
        for tc, task in pending:
            try:
                result = await task
            except Exception as e:
                logger.warning("Tool %s failed: %s", tc.name, e)
                result = f"Error executing {tc.name}: {e}"
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        return None  # Signal: tool calls were executed

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _shadow_head(self) -> str | None:
        """Return the current shadow repo HEAD commit hash, or None."""
        try:
            return self.shadow._git("rev-parse", "HEAD")
        except Exception:
            return None
