"""LLM data types ? ToolCall, LLMResponse, SSEEvent.

These are pure data structures shared between the LLM client and the
agent loop.  Separating them from the client prevents circular imports
and keeps the type system lightweight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class SSEEvent:
    """A single event in the SSE (Server-Sent Events) stream.

    Types:
    - "text": a content token
    - "reasoning": a reasoning token (DeepSeek/o1)
    - "tool_call": a complete tool call (args parsed as valid JSON)
    - "done": stream finished
    - "error": stream error
    """

    type: str
    token: str | None = None
    tool_call: ToolCall | None = None
    usage: dict | None = None
    error: str | None = None


@dataclass
class LLMResponse:
    content: str = ""
    reasoning_content: str = ""  # thinking tokens
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        # DeepSeek requires reasoning_content to be passed back in subsequent
        # messages, otherwise it returns a 400 error.
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg
