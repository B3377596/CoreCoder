"""Context assembly pipeline.

The pipeline transforms a ContextRequest into a ContextBundle through
a series of modular stages:

    collect_candidates → rank → deduplicate → compress → budget_trim → assemble

Each stage is a pure function from state to state, making the pipeline
testable and composable.  Stages can be extended, reordered, or replaced
without affecting the rest of the system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextBundle,
    ContextRequest,
    TokenBudget,
    ExecutionState,
    ContextSource,
    ContextType,
)
from corecoder.orchestration.context.ranker import ContextRanker


# ===========================================================================
# Pipeline context — mutable state flowing through stages
# ===========================================================================

@dataclass
class PipelineState:
    """Mutable state that flows through the pipeline stages."""

    request: ContextRequest
    budget: TokenBudget
    fragments: list[ContextFragment] = field(default_factory=list)
    discarded: list[ContextFragment] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)


# ===========================================================================
# Stage type
# ===========================================================================

PipelineStage = Callable[[PipelineState], PipelineState]


# ===========================================================================
# Pipeline
# ===========================================================================

class ContextAssemblyPipeline:
    """Orchestrates the stages of context assembly.

    Each stage mutates the PipelineState and returns it, enabling
    chaining and easy insertion of new stages.
    """

    def __init__(self, ranker: ContextRanker | None = None):
        self._ranker = ranker or ContextRanker()
        self._stages: list[tuple[str, PipelineStage]] = [
            ("rank", self._stage_rank),
            ("deduplicate", self._stage_deduplicate),
            ("compress", self._stage_compress),
            ("budget_trim", self._stage_budget_trim),
        ]

    def add_stage(self, name: str, stage: PipelineStage, after: str = "") -> None:
        """Insert a custom stage into the pipeline."""
        if after:
            for i, (sn, _) in enumerate(self._stages):
                if sn == after:
                    self._stages.insert(i + 1, (name, stage))
                    return
        self._stages.append((name, stage))

    def run(
        self,
        fragments: list[ContextFragment],
        request: ContextRequest,
        budget: TokenBudget | None = None,
    ) -> ContextBundle:
        """Run the full assembly pipeline."""
        budget = budget or TokenBudget.default()
        state = PipelineState(request=request, budget=budget, fragments=list(fragments))

        t0 = time.time()

        for stage_name, stage_fn in self._stages:
            t_stage = time.time()
            state = stage_fn(state)
            state.stage_timings[stage_name] = (time.time() - t_stage) * 1000.0

        assembly_time_ms = (time.time() - t0) * 1000.0

        # Compute token usage
        token_usage: dict[str, int] = {}
        total_tokens = 0
        for f in state.fragments:
            layer = f.source.value
            token_usage[layer] = token_usage.get(layer, 0) + f.token_count
            total_tokens += f.token_count

        # Compression ratio: output tokens / input tokens
        input_tokens = sum(f.token_count for f in fragments)
        compression_ratio = (
            (1.0 - total_tokens / input_tokens) if input_tokens > 0 else 0.0
        )

        return ContextBundle(
            fragments=state.fragments,
            token_usage=token_usage,
            total_tokens_used=total_tokens,
            budget=budget,
            compression_ratio=compression_ratio,
            assembly_time_ms=assembly_time_ms,
            discarded_fragments=state.discarded,
            metadata={
                "stage_timings": state.stage_timings,
                "input_fragment_count": len(fragments),
                "output_fragment_count": len(state.fragments),
                "discarded_count": len(state.discarded),
            },
        )

    # ------------------------------------------------------------------
    # Stage 1: Rank
    # ------------------------------------------------------------------

    def _stage_rank(self, state: PipelineState) -> PipelineState:
        """Score and sort fragments by relevance."""
        if not state.fragments:
            return state
        state.fragments = self._ranker.rank(state.fragments, state.request)
        state.stats["ranked_count"] = len(state.fragments)
        return state

    # ------------------------------------------------------------------
    # Stage 2: Deduplicate
    # ------------------------------------------------------------------

    def _stage_deduplicate(self, state: PipelineState) -> PipelineState:
        """Remove duplicate and near-duplicate fragments.

        Uses content hashing and fuzzy matching to detect duplicates.
        When duplicates are found, keeps the one with the higher score.
        """
        if not state.fragments:
            return state

        seen_hashes: set[int] = set()
        seen_files: set[str] = set()
        deduped: list[ContextFragment] = []

        for f in state.fragments:
            # Exact hash check
            content_hash = hash(f.content[:500])  # First 500 chars as fingerprint
            if content_hash in seen_hashes:
                # Duplicate — keep the one with higher score
                existing = next((df for df in deduped if hash(df.content[:500]) == content_hash), None)
                if existing and f.effective_score > existing.effective_score:
                    deduped.remove(existing)
                    state.discarded.append(existing)
                    deduped.append(f)
                else:
                    state.discarded.append(f)
                continue

            # File-level deduplication: don't include the same file twice
            if f.origin_file:
                if f.origin_file in seen_files:
                    state.discarded.append(f)
                    continue
                seen_files.add(f.origin_file)

            seen_hashes.add(content_hash)
            deduped.append(f)

        removed = len(state.fragments) - len(deduped)
        state.stats["deduplicated_count"] = removed
        state.fragments = deduped
        return state

    # ------------------------------------------------------------------
    # Stage 3: Compress
    # ------------------------------------------------------------------

    def _stage_compress(self, state: PipelineState) -> PipelineState:
        """Intelligently compress fragments that exceed their size limits.

        Compression strategies by fragment type:
        - CODE: keep first N lines, add line count note
        - TOOL_RESULT: truncate middle, keep head and tail
        - SUMMARY: keep as-is (already compressed)
        - ERROR: keep full stack trace
        """
        for f in state.fragments:
            if f.max_tokens <= 0:
                continue

            est_tokens = f.token_count
            if est_tokens <= f.max_tokens:
                continue

            # Need to compress
            ratio = f.max_tokens / max(1, est_tokens)

            if f.type == ContextType.CODE:
                f.content = self._compress_code(f.content, f.max_tokens)
            elif f.type == ContextType.OUTPUT:
                f.content = self._compress_output(f.content, f.max_tokens)
            elif f.type == ContextType.ERROR:
                # Keep errors intact if possible
                if est_tokens > f.max_tokens * 2:
                    f.content = self._compress_output(f.content, f.max_tokens)
            else:
                # Generic: truncate with marker
                f.content = self._compress_generic(f.content, f.max_tokens)

            f.token_count = max(1, len(f.content) // 3)
            f.metadata["compressed"] = True
            f.metadata["compression_ratio"] = ratio

        return state

    def _compress_code(self, content: str, max_tokens: int) -> str:
        """Compress code: keep first N lines, add summary."""
        char_limit = max_tokens * 3
        lines = content.split("\n")
        if len(lines) <= char_limit // 40:  # ~40 chars per line average
            return content

        keep = max(10, char_limit // 40)
        header = "\n".join(lines[:keep])
        return f"{header}\n... ({len(lines) - keep} lines trimmed)"

    def _compress_output(self, content: str, max_tokens: int) -> str:
        """Compress tool output: keep head and tail, snip middle."""
        char_limit = max_tokens * 3
        if len(content) <= char_limit:
            return content

        head_size = char_limit // 2
        tail_size = char_limit // 4
        head = content[:head_size]
        tail = content[-tail_size:]
        return f"{head}\n... ({len(content) - head_size - tail_size} chars snipped) ...\n{tail}"

    def _compress_generic(self, content: str, max_tokens: int) -> str:
        """Generic truncation."""
        char_limit = max_tokens * 3
        if len(content) <= char_limit:
            return content
        return content[:char_limit] + "\n... (truncated)"

    # ------------------------------------------------------------------
    # Stage 4: Budget trim
    # ------------------------------------------------------------------

    def _stage_budget_trim(self, state: PipelineState) -> PipelineState:
        """Trim fragments to fit within the token budget.

        Fragments are already sorted by effective_score (from ranking).
        We allocate tokens per layer based on the budget, filling
        high-priority layers first.
        """
        budget = state.budget
        if not budget.layers:
            return state

        # Track tokens used per layer
        used: dict[str, int] = {}
        kept: list[ContextFragment] = []

        for f in state.fragments:
            layer_name = f.source.value
            layer_limit = budget.get_budget(layer_name)

            # If no explicit limit for this layer, use a proportional share
            if layer_limit == 0:
                layer_limit = budget.total_budget // max(1, len(budget.layers))

            current_usage = used.get(layer_name, 0)

            if current_usage + f.token_count <= layer_limit:
                kept.append(f)
                used[layer_name] = current_usage + f.token_count
            else:
                # Try to still include high-priority fragments even if over budget
                if f.priority >= 9:
                    kept.append(f)
                    used[layer_name] = current_usage + f.token_count
                else:
                    state.discarded.append(f)

        state.stats["budget_trimmed"] = len(state.fragments) - len(kept)
        state.fragments = kept
        return state
