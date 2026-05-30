"""LLM provider layer ? thin wrapper over OpenAI-compatible APIs.

Since most providers expose an OpenAI-compatible endpoint, we use the
openai SDK directly.  For non-OpenAI providers, use the LiteLLM backend.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from typing import AsyncGenerator

from openai import (
    AsyncOpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)

from corecoder.llm.types import ToolCall, SSEEvent, LLMResponse


# pricing per million tokens: (input, output)
_PRICING = {
    # OpenAI
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
}

# Disabled by default: dumping prompts/tool calls is useful locally but
# too noisy and risky for everyday use


class LLM:
    """Async LLM client for OpenAI-compatible APIs.

    All chat() calls are async.  Tool calls, retries, and stream processing
    happen without blocking the event loop.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def estimated_cost(self) -> float | None:
        pricing = _PRICING.get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return (
            self.total_prompt_tokens * input_rate / 1_000_000
            + self.total_completion_tokens * output_rate / 1_000_000
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls."""
        params = self._build_params(messages, tools)
        # stream_options is an OpenAI extension; not all providers support it
        try:
            params["stream_options"] = {"include_usage": True}
            stream = await self._create_stream_with_retry(params)
        except Exception:
            params.pop("stream_options", None)
            stream = await self._create_stream_with_retry(params)

        return await self._process_stream(stream, on_token)

    async def chat_sse(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Stream LLM response as SSE events.

        Unlike chat() which waits for the full response, this yields events
        as soon as they're available:
        - "text" events for each content token (caller can render immediately)
        - "reasoning" events for thinking tokens
        - "tool_call" events as soon as each tool's arguments form valid JSON
          (caller can start tool execution immediately, overlapping with
          remaining stream processing)
        - "done" event when the stream completes

        This enables the agent to start executing tools before the LLM
        finishes generating other tool calls or trailing text.
        """
        params = self._build_params(messages, tools)
        try:
            params["stream_options"] = {"include_usage": True}
            stream = await self._create_stream_with_retry(params)
        except Exception:
            params.pop("stream_options", None)
            stream = await self._create_stream_with_retry(params)

        async for event in self._process_stream_sse(stream):
            yield event

    # ------------------------------------------------------------------
    # stream creation (override point for LiteLLM)
    # ------------------------------------------------------------------

    def _build_params(self, messages: list[dict], tools: list[dict] | None) -> dict:
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools
        return params

    async def _create_stream_with_retry(self, params: dict, max_retries: int = 3):
        """Call OpenAI-compatible API with retry on transient errors."""
        for attempt in range(max_retries):
            try:
                return await self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except APIError as e:
                # 5xx = server error, retry; 4xx = client error, don't
                if e.status_code and e.status_code >= 500 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # ------------------------------------------------------------------
    # SSE stream processing ? yields tool calls as soon as they're complete
    # ------------------------------------------------------------------

    async def _process_stream_sse(self, stream) -> AsyncGenerator[SSEEvent, None]:
        """Process a stream incrementally, yielding SSE events as they arrive.

        Tool calls are yielded as soon as their accumulated arguments parse
        as valid JSON.  This lets the caller start executing a tool while
        the LLM is still generating other tool calls in the same response.

        Args passed across multiple chunks are accumulated; we try parsing
        after each chunk.  Once a tool call parses successfully, it's marked
        "yielded" and won't be re-emitted.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, args, yielded}
        prompt_tok = 0
        completion_tok = 0

        async for chunk in stream:
            # ---- usage info ----
            usage = chunk.usage
            if usage:
                if isinstance(usage, dict):
                    prompt_tok = usage.get("prompt_tokens", 0) or 0
                    completion_tok = usage.get("completion_tokens", 0) or 0
                else:
                    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # ---- reasoning tokens ----
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                yield SSEEvent(type="reasoning", token=reasoning)

            # ---- content tokens ----
            if delta.content:
                content_parts.append(delta.content)
                yield SSEEvent(type="text", token=delta.content)

            # ---- tool call deltas ----
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": "", "yielded": False}
                    entry = tc_map[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["args"] += tc_delta.function.arguments

                # After accumulating deltas for this chunk, check if any
                # tool call's arguments now form valid JSON.  If so, that
                # tool call is complete ? yield it immediately.
                for idx in sorted(tc_map):
                    entry = tc_map[idx]
                    if entry["yielded"] or not entry["name"] or not entry["args"]:
                        continue
                    try:
                        args = json.loads(entry["args"])
                    except json.JSONDecodeError:
                        continue  # Still being built ? wait for more chunks
                    # Valid JSON ? tool call is complete
                    entry["yielded"] = True
                    yield SSEEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=entry["id"],
                            name=entry["name"],
                            arguments=args,
                        ),
                    )

        # ---- stream complete ----
        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        # Emit any tool calls that didn't parse (malformed JSON) as a
        # best-effort fallback ? use empty args
        for idx in sorted(tc_map):
            entry = tc_map[idx]
            if not entry["yielded"] and entry["name"]:
                yield SSEEvent(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=entry["id"],
                        name=entry["name"],
                        arguments={},
                    ),
                )

        yield SSEEvent(
            type="done",
            usage={
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
            },
        )

    # ------------------------------------------------------------------
    # shared stream processing (used by both LLM and LiteLLM)
    # ------------------------------------------------------------------

    async def _process_stream(self, stream, on_token=None) -> LLMResponse:
        """Parse a streaming response into an LLMResponse.

        Works with both OpenAI and LiteLLM stream objects (they share the
        same structural conventions for choices/delta/tool_calls/usage).
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, args}
        prompt_tok = 0
        completion_tok = 0

        async for chunk in stream:
            # usage info (final chunk for OpenAI; may appear earlier for some providers)
            usage = chunk.usage
            if usage:
                if isinstance(usage, dict):
                    prompt_tok = usage.get("prompt_tokens", 0) or 0
                    completion_tok = usage.get("completion_tokens", 0) or 0
                else:
                    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # DeepSeek / o1 reasoning tokens ? must be passed back to API
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

            # accumulate text
            if delta.content:
                content_parts.append(delta.content)
                if on_token:
                    ret = on_token(delta.content)
                    if inspect.isawaitable(ret):
                        await ret

            # accumulate tool calls across chunks
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        return LLMResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )


class LiteLLM(LLM):
    """Async LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an AsyncOpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _build_params(self, messages: list[dict], tools: list[dict] | None) -> dict:
        params = super()._build_params(messages, tools)
        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url
        return params

    async def _create_stream_with_retry(self, params: dict, max_retries: int = 3):
        """Call litellm with retry on transient errors."""
        import litellm

        for attempt in range(max_retries):
            try:
                return await litellm.acompletion(**params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
