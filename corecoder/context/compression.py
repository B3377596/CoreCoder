"""Multi-layer context compression.

Claude Code uses a 4-layer strategy:
  1. HISTORY_SNIP      - trim old tool outputs to a one-line summary
  2. Microcompact       - LLM-powered summary of old turns (cached)
  3. CONTEXT_COLLAPSE   - aggressive compression when nearing hard limit
  4. Autocompact        - periodic background compaction

CoreCoder implements the same idea in 3 layers:
  Layer 1 (tool_snip)     - replace verbose tool results with truncated versions
  Layer 2 (summarize)     - LLM-powered summary of old conversation
  Layer 3 (hard_collapse) - last resort: drop everything except summary + recent

Token counting uses ``tiktoken`` when available for accurate counts,
falling back to a character-based estimate.
"""
from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.client import LLM

logger = logging.getLogger("corecoder.context")

# ------------------------------------------------------------------
# token counting
# ------------------------------------------------------------------
_tiktoken_enc = None

def _get_encoder():
    """Lazy-load tiktoken cl100k_base encoder (covers GPT-4, Claude, most models)."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _tiktoken_enc = False
    return _tiktoken_enc if _tiktoken_enc is not False else None

def count_tokens(text: str) -> int:
    """Count tokens accurately with tiktoken, or fall back to estimate."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    return len(text) // 3

def estimate_tokens(messages: list[dict]) -> int:
    """Count total tokens across all messages."""
    total = 0
    for m in messages:
        if m.get("content"):
            total += count_tokens(m["content"])
        if m.get("tool_calls"):
            total += count_tokens(str(m["tool_calls"]))
    return total

# ------------------------------------------------------------------
# context manager
# ------------------------------------------------------------------
class ContextManager:
    def __init__(self, max_tokens: int = 256_000):
        self.max_tokens = max_tokens
        self._snip_at = int(max_tokens * 0.50)
        self._summarize_at = int(max_tokens * 0.70)
        self._collapse_at = int(max_tokens * 0.90)

    async def maybe_compress(
        self, messages: list[dict], llm: LLM | None = None,
        ephemeral_overhead: int = 0,
    ) -> bool:
        """Apply compression layers as needed. Returns True if compression happened.

        Args:
            messages: The persistent_history list (NOT including ephemeral context).
            llm: LLM client for summarization.
            ephemeral_overhead: Estimated token count of the ephemeral prefix
                (system + assistant(mem) + assistant(repo) + assistant(run)).
                Added to message token count so compression thresholds
                account for total context size, not just persistent history.
        """
        current = estimate_tokens(messages) + ephemeral_overhead
        compressed = False
        # Layer 1: snip verbose tool outputs
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)
                logger.debug("Layer 1 (snip) applied: %d tokens", current)
        # Layer 2: LLM-powered summarization of old turns
        if current > self._summarize_at and len(messages) > 10:
            if await self._summarize_old(messages, llm, keep_recent=8):
                compressed = True
                current = estimate_tokens(messages)
                logger.debug("Layer 2 (summarize) applied: %d tokens", current)
        # Layer 3: hard collapse - last resort
        if current > self._collapse_at and len(messages) > 4:
            await self._hard_collapse(messages, llm)
            compressed = True
            logger.debug("Layer 3 (hard collapse) applied: %d tokens", estimate_tokens(messages))
        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """Layer 1: Truncate tool results over 1500 chars to first/last lines."""
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            snipped = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            m["content"] = snipped
            changed = True
        return changed

    async def _summarize_old(self, messages: list[dict], llm: LLM | None,
                             keep_recent: int = 8) -> bool:
        """Layer 2: Summarize old conversation, keep recent messages intact."""
        if len(messages) <= keep_recent:
            return False
        old = messages[:-keep_recent]
        tail = messages[-keep_recent:]
        summary = await self._get_summary(old, llm)
        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(tail)
        return True

    async def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """Layer 3: Emergency compression. Keep only last 4 messages + summary."""
        tail = messages[-4:] if len(messages) > 4 else messages[-2:]
        summary = await self._get_summary(messages[:-len(tail)], llm)
        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)

    async def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fallback to extraction."""
        flat = self._flatten(messages)
        if llm:
            try:
                resp = await llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception as e:
                logger.warning("LLM summarization failed, using fallback: %s", e)
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "*")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        files_seen: set[str] = set()
        errors: list[str] = []
        for m in messages:
            text = m.get("content", "") or ""
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            for line in text.splitlines():
                if 'error' in line.lower() or 'Error' in line:
                    errors.append(line.strip()[:150])
        parts: list[str] = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
