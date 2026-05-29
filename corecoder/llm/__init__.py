"""LLM client interface ? OpenAI-compatible + LiteLLM backends."""

from corecoder.llm.types import ToolCall, SSEEvent, LLMResponse
from corecoder.llm.client import LLM, LiteLLM

__all__ = ["ToolCall", "SSEEvent", "LLMResponse", "LLM", "LiteLLM"]
