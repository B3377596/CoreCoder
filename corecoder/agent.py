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
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING

from .tools import ALL_TOOLS, get_tool
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager

if TYPE_CHECKING:
    from .llm import LLM

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
        self._system = system_prompt(self.tools)
        # checkpoint stack: (messages_length, description)
        self._checkpoints: list[tuple[int, str]] = []

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._checkpoint(f"user: {user_input[:60]}")

        self.messages.append({"role": "user", "content": user_input})
        await self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = await self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )

            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            self.messages.append(resp.message)

            if len(resp.tool_calls) == 1:
                tc = resp.tool_calls[0]
                if on_tool:
                    on_tool(tc.name, tc.arguments)
                result = await self._call_tool(tc)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            else:
                results = await self._call_tools_parallel(resp.tool_calls, on_tool)
                for tc, result in zip(resp.tool_calls, results):
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            await self.context.maybe_compress(self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    def reset(self):
        """Clear conversation history and checkpoints."""
        self.messages.clear()
        self._checkpoints.clear()

    # ------------------------------------------------------------------
    # checkpoint / undo
    # ------------------------------------------------------------------

    def _checkpoint(self, description: str = ""):
        self._checkpoints.append((len(self.messages), description))

    def undo(self) -> str | None:
        """Restore to before the last user turn. Returns the undone description."""
        if not self._checkpoints:
            return None

        self._checkpoints.pop()  # discard current state

        if not self._checkpoints:
            self.messages.clear()
            return "(returned to initial state)"

        target_len, desc = self._checkpoints[-1]
        self.messages = self.messages[:target_len]
        return desc

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------

    async def _call_tool(self, tc) -> str:
        """Execute a single tool call. Routes sync tools to a thread pool
        and runs async tools directly on the event loop."""
        tool = get_tool(tc.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tc.name)
            return f"Error: unknown tool '{tc.name}'"
        return await self._invoke(tool, tc.arguments)

    async def _call_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Execute multiple tool calls concurrently.

        Each sync tool runs in its own thread (via ``asyncio.to_thread``)
        so they execute in parallel without blocking the event loop.
        Async tools run directly on the loop.
        """
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
        """Invoke a tool, handling both sync and async implementations."""
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
    # helpers
    # ------------------------------------------------------------------

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]
