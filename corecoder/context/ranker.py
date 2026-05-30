"""Relevance scoring engine for context fragments.

Ranks context fragments based on multiple signals:
- Semantic similarity (keyword/token overlap)
- Repository graph distance (dependency proximity)
- Symbol overlap
- Task relevance
- Recency
- Execution state match
- Failure similarity

Architecture supports future embedding-based scoring via the Scorer interface.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from corecoder.context.models import (
    ContextFragment,
    ContextType,
    ContextSource,
    ExecutionState,
    ContextRequest,
)


# ===========================================================================
# Scoring function registry  each scorer returns [0.0, 1.0]
# ===========================================================================

@dataclass
class ScoreComponent:
    """A single dimension of relevance scoring."""

    name: str
    weight: float  # How much this component contributes to the final score
    score: float = 0.0
    explanation: str = ""


class ContextRanker:
    """Ranks context fragments by relevance to the current execution state.

    The ranker combines multiple scoring functions into a weighted composite
    score.  Each scorer is a simple, fast heuristic  no external API calls.

    For production use with embeddings, subclass and override _score_semantic().
    """

    def __init__(self):
        self._scorers: list[tuple[str, float]] = [
            ("semantic", 0.30),
            ("task_relevance", 0.20),
            ("source_priority", 0.15),
            ("recency", 0.10),
            ("state_match", 0.10),
            ("error_relevance", 0.10),
            ("anti_noise_penalty", 0.05),
        ]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def rank(
        self,
        fragments: list[ContextFragment],
        request: ContextRequest,
    ) -> list[ContextFragment]:
        """Score and sort fragments by relevance.

        Returns a new list with updated relevance_score on each fragment.
        Original fragments are not mutated.
        """
        if not fragments:
            return []

        # Build the query context for scoring
        query = self._build_query(request)

        # Score each fragment
        scored: list[ContextFragment] = []
        for f in fragments:
            components = self._score_fragment(f, query, request)
            composite = sum(c.score * c.weight for c in components)
            # Clamp to [0, 1]
            composite = max(0.0, min(1.0, composite))
            # Create scored copy
            scored.append(ContextFragment(
                id=f.id,
                source=f.source,
                type=f.type,
                content=f.content,
                relevance_score=composite,
                priority=f.priority,
                confidence=f.confidence,
                token_count=f.token_count,
                max_tokens=f.max_tokens,
                timestamp=f.timestamp,
                ttl=f.ttl,
                origin_task_id=f.origin_task_id,
                origin_file=f.origin_file,
                metadata={**f.metadata, "score_components": {
                    c.name: c.score for c in components
                }},
            ))

        # Sort by effective score descending
        scored.sort(key=lambda f: f.effective_score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # query construction
    # ------------------------------------------------------------------

    def _build_query(self, request: ContextRequest) -> dict[str, Any]:
        """Build a query descriptor from the context request."""
        return {
            "task_title": request.task_title,
            "task_description": request.task_description,
            "goal": request.goal,
            "execution_state": request.execution_state,
            "focus_files": request.focus_files,
            "focus_symbols": request.focus_symbols,
            "recent_errors": request.recent_errors,
            "constraints": request.constraints,
            "keywords": _extract_keywords(
                request.task_title + " " + request.task_description + " " + request.goal
            ),
        }

    # ------------------------------------------------------------------
    # per-fragment scoring
    # ------------------------------------------------------------------

    def _score_fragment(
        self,
        fragment: ContextFragment,
        query: dict[str, Any],
        request: ContextRequest,
    ) -> list[ScoreComponent]:
        """Run all scorers on a single fragment."""
        return [
            self._score_semantic(fragment, query),
            self._score_task_relevance(fragment, query),
            self._score_source_priority(fragment),
            self._score_recency(fragment),
            self._score_state_match(fragment, request),
            self._score_error_relevance(fragment, query),
            self._score_anti_noise(fragment),
        ]

    # ------------------------------------------------------------------
    # individual scorers
    # ------------------------------------------------------------------

    def _score_semantic(
        self, fragment: ContextFragment, query: dict[str, Any]
    ) -> ScoreComponent:
        """Keyword-overlap based semantic similarity.

        Fast and language-agnostic.  Replace with embedding cosine similarity
        for production use.
        """
        keywords: list[str] = query.get("keywords", [])
        if not keywords:
            return ScoreComponent("semantic", 0.30, 0.5)

        content_lower = fragment.content.lower()
        hits = sum(1 for kw in keywords if kw.lower() in content_lower)
        score = min(1.0, hits / max(1, len(keywords)) * 1.5)
        return ScoreComponent("semantic", 0.30, score,
                              f"{hits}/{len(keywords)} keywords matched")

    def _score_task_relevance(
        self, fragment: ContextFragment, query: dict[str, Any]
    ) -> ScoreComponent:
        """How closely does this fragment relate to the current task*"""
        # Fragments from completed dependencies are highly relevant
        if fragment.source == ContextSource.ARTIFACT:
            return ScoreComponent("task_relevance", 0.20, 0.9, "dependency artifact")
        if fragment.source == ContextSource.TASK:
            return ScoreComponent("task_relevance", 0.20, 1.0, "current task")
        if fragment.source == ContextSource.WORKING_MEMORY:
            return ScoreComponent("task_relevance", 0.20, 0.85, "working memory")
        if fragment.source == ContextSource.CONSTRAINT:
            return ScoreComponent("task_relevance", 0.20, 0.9, "constraint")
        if fragment.type == ContextType.ERROR:
            return ScoreComponent("task_relevance", 0.20, 0.8, "error context")
        return ScoreComponent("task_relevance", 0.20, 0.4, "general")

    def _score_source_priority(self, fragment: ContextFragment) -> ScoreComponent:
        """Normalize the fragment's priority into a [0,1] score."""
        return ScoreComponent("source_priority", 0.15, fragment.priority / 10.0)

    def _score_recency(self, fragment: ContextFragment) -> ScoreComponent:
        """More recent fragments score higher."""
        age_s = time.time() - fragment.timestamp
        if age_s < 60:
            score = 1.0
        elif age_s < 300:
            score = 0.8
        elif age_s < 900:
            score = 0.5
        else:
            score = 0.2
        return ScoreComponent("recency", 0.10, score, f"age={age_s:.0f}s")

    def _score_state_match(
        self, fragment: ContextFragment, request: ContextRequest
    ) -> ScoreComponent:
        """Adjust score based on execution state.

        Different states value different fragment types differently.
        """
        state = request.execution_state

        # Coding: value code and symbol fragments highly
        if state == ExecutionState.CODING:
            if fragment.type in (ContextType.CODE, ContextType.SYMBOL_DEF):
                return ScoreComponent("state_match", 0.10, 1.0, "coding needs code")
            if fragment.source == ContextSource.REPOSITORY:
                return ScoreComponent("state_match", 0.10, 0.9, "coding needs repo")

        # Debugging: value error fragments highly
        if state == ExecutionState.DEBUGGING:
            if fragment.type == ContextType.ERROR:
                return ScoreComponent("state_match", 0.10, 1.0, "debugging needs errors")
            if fragment.source == ContextSource.FAILURE_MEMORY:
                return ScoreComponent("state_match", 0.10, 0.95, "debugging needs failures")

        # Planning: value repository overview highly
        if state == ExecutionState.PLANNING:
            if fragment.source == ContextSource.REPOSITORY:
                return ScoreComponent("state_match", 0.10, 1.0, "planning needs overview")
            if fragment.source == ContextSource.SYMBOL:
                return ScoreComponent("state_match", 0.10, 0.9, "planning needs symbols")

        # Testing/Verifying: value tool results
        if state in (ExecutionState.TESTING, ExecutionState.VERIFYING):
            if fragment.source == ContextSource.TOOL_RESULT:
                return ScoreComponent("state_match", 0.10, 1.0, "verifying needs results")

        return ScoreComponent("state_match", 0.10, 0.5, "default")

    def _score_error_relevance(
        self, fragment: ContextFragment, query: dict[str, Any]
    ) -> ScoreComponent:
        """Boost fragments related to recent errors."""
        recent_errors: list[str] = query.get("recent_errors", [])
        if not recent_errors:
            return ScoreComponent("error_relevance", 0.10, 0.0, "no recent errors")

        # Check if fragment content overlaps with error messages
        content_lower = fragment.content.lower()
        hits = 0
        for err in recent_errors[:3]:
            # Extract key terms from the error
            err_terms = set(re.findall(r'\w+', err.lower()))
            if err_terms:
                overlap = sum(1 for t in err_terms if t in content_lower)
                if overlap > len(err_terms) * 0.3:
                    hits += 1

        if hits == 0:
            return ScoreComponent("error_relevance", 0.10, 0.0)
        score = min(1.0, hits / 3.0 * 1.5)
        return ScoreComponent("error_relevance", 0.10, score,
                              f"{hits} error overlaps")

    def _score_anti_noise(self, fragment: ContextFragment) -> ScoreComponent:
        """Penalize fragments likely to be noise.

        - Duplicate content
        - Very short fragments with no actionable information
        - Generic/boilerplate content
        - Empty fragments
        """
        if not fragment.content.strip():
            return ScoreComponent("anti_noise", 0.05, -0.5, "empty fragment")

        # Penalize very short fragments (likely not useful alone)
        if len(fragment.content) < 20:
            return ScoreComponent("anti_noise", 0.05, -0.2, "very short")

        # Penalize fragments that are just a single line of punctuation/whitespace
        cleaned = fragment.content.strip()
        if len(set(cleaned)) < 5:
            return ScoreComponent("anti_noise", 0.05, -0.3, "low entropy")

        return ScoreComponent("anti_noise", 0.05, 0.0, "normal")


# ===========================================================================
# helpers
# ===========================================================================

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from task text.

    Splits on camelCase, snake_case, and natural language word boundaries.
    Filters out common stop words.
    """
    if not text:
        return []

    _STOP_WORDS = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "shall", "this", "that", "these", "those",
        "it", "its", "we", "you", "they", "he", "she", "not", "no", "but",
        "with", "from", "by", "as", "if", "then", "than", "so", "also",
    }

    # Split on camelCase and snake_case and whitespace/punctuation
    words = re.findall(
        r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]+|\d+",
        text,
    )
    keywords = []
    for w in words:
        wl = w.lower()
        if len(wl) > 2 and wl not in _STOP_WORDS:
            keywords.append(w)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique.append(k)
    return unique[:30]  # Cap at 30 keywords
