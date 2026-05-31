"""Core data models for the Context Orchestrator.

Every piece of context flowing through the system is a ContextFragment ?
a typed, scored, timestamped atom.  This design enforces that context is
never a raw string; it always carries provenance, relevance, and budget
metadata so the pipeline can make informed filtering decisions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Context source ? where did this fragment come from*
# ---------------------------------------------------------------------------

class ContextSource(str, Enum):
    """Provenance tag for every context fragment.

    The source determines default priority and compression policy.
    """

    SYSTEM = "system"               # System prompt, global instructions
    TASK = "task"                    # Current stage task / objective
    WORKING_MEMORY = "working_memory"  # In-flight execution state
    REPOSITORY = "repository"       # Files, directory structure
    SYMBOL = "symbol"               # Functions, classes, types
    DEPENDENCY_GRAPH = "dependency_graph"  # Import/call graph neighborhood
    TOOL_RESULT = "tool_result"     # Output from a tool invocation
    FAILURE_MEMORY = "failure_memory"  # Past errors and root causes
    CONVERSATION_SUMMARY = "conversation_summary"  # Compressed history
    CONSTRAINT = "constraint"       # Explicit rules/limits
    ARTIFACT = "artifact"           # Outputs from completed tasks
    PLAN = "plan"                   # Task graph structure, plan overview
    USER = "user"                   # User-provided context or hints


# ---------------------------------------------------------------------------
# Context type ? what kind of information is this*
# ---------------------------------------------------------------------------

class ContextType(str, Enum):
    """Semantic category of the fragment content."""

    INSTRUCTION = "instruction"      # Directive, rule, or constraint
    CODE = "code"                    # Source code content
    SYMBOL_DEF = "symbol_def"        # Function/class/type definition
    SYMBOL_REF = "symbol_ref"        # Reference to a symbol
    DEPENDENCY = "dependency"        # Import or call relationship
    ERROR = "error"                  # Error message or stack trace
    SUMMARY = "summary"              # Summarized/compressed information
    OUTPUT = "output"                # Execution or tool output
    METADATA = "metadata"            # File stats, git info, etc.
    CONSTRAINT = "constraint"        # Explicit limit or rule
    ARTIFACT = "artifact"            # Completed work product
    PLAN_NODE = "plan_node"          # A task in the plan graph


# ---------------------------------------------------------------------------
# Execution state ? what phase is the agent currently in*
# ---------------------------------------------------------------------------

class ExecutionState(str, Enum):
    """The current phase of task execution.

    Different states receive radically different context profiles.
    See policies.py for the mapping.
    """

    PLANNING = "planning"        # Decomposing goal into tasks
    EXPLORING = "exploring"      # Understanding the codebase
    CODING = "coding"            # Writing/editing files
    TESTING = "testing"          # Running tests, validating
    DEBUGGING = "debugging"      # Investigating failures
    VERIFYING = "verifying"      # Checking completion criteria
    REFACTORING = "refactoring"  # Restructuring existing code


# ---------------------------------------------------------------------------
# ContextFragment ? the atomic unit of context
# ---------------------------------------------------------------------------

@dataclass
class ContextFragment:
    """A single piece of context flowing through the pipeline.

    Every fragment carries:
    - Identity (id, source, type)
    - Content (the actual text/symbol/code)
    - Scoring metadata (relevance_score, priority, confidence)
    - Budget metadata (token_count)
    - Temporal metadata (timestamp, ttl)
    - Debug metadata (metadata dict)

    Fragments are immutable after creation.  The pipeline creates new
    fragments rather than mutating existing ones.
    """

    id: str = field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:8]}")
    source: ContextSource = ContextSource.SYSTEM
    type: ContextType = ContextType.INSTRUCTION
    content: str = ""

    # Relevance scoring
    relevance_score: float = 0.5       # 0.0?1.0, higher = more relevant
    priority: int = 5                  # 1?10, higher = keep when budget is tight
    confidence: float = 1.0            # 0.0?1.0, how confident the scorer is

    # Token budget
    token_count: int = 0               # Estimated token count of content
    max_tokens: int = 0                # Optional cap for this fragment

    # Temporal
    timestamp: float = field(default_factory=time.time)
    ttl: float = 0.0                   # 0 = never expires; >0 = seconds until stale

    # Provenance
    origin_task_id: str = ""           # Which task generated this
    origin_file: str = ""             # Which file (for repo fragments)

    # Debug / extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stale(self) -> bool:
        """Check if this fragment has exceeded its TTL."""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @property
    def effective_score(self) -> float:
        """Composite score blending relevance, priority, and confidence."""
        return self.relevance_score * 0.6 + (self.priority / 10.0) * 0.25 + self.confidence * 0.15


# ---------------------------------------------------------------------------
# TokenBudget ? hierarchical token allocation
# ---------------------------------------------------------------------------

@dataclass
class LayerBudget:
    """Token allocation for a single context layer."""

    layer_name: str
    max_tokens: int
    min_tokens: int = 0
    priority: int = 5  # Higher-priority layers get budget first during trimming

    def __post_init__(self):
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be >= min_tokens ({self.min_tokens})"
            )


@dataclass
class TokenBudget:
    """Hierarchical token budget across all context layers.

    The total budget is divided among layers.  When the budget is exceeded,
    fragments are trimmed from the lowest-priority layers first.

    Percentages should sum to <= 100%.  The remaining is reserved for
    the system prompt and response.
    """

    total_budget: int = 100_000  # Total token budget for context
    layers: dict[str, LayerBudget] = field(default_factory=dict)

    # Convenience: define by percentages of total_budget
    layer_percentages: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.layer_percentages and not self.layers:
            for name, pct in self.layer_percentages.items():
                self.layers[name] = LayerBudget(
                    layer_name=name,
                    max_tokens=int(self.total_budget * pct / 100.0),
                )

    def get_budget(self, layer_name: str) -> int:
        """Get the token budget for a specific layer."""
        lb = self.layers.get(layer_name)
        return lb.max_tokens if lb else 0

    def remaining_budget(self, used: dict[str, int]) -> dict[str, int]:
        """Calculate remaining token budget per layer."""
        remaining: dict[str, int] = {}
        for name, lb in self.layers.items():
            remaining[name] = max(0, lb.max_tokens - used.get(name, 0))
        return remaining

    @staticmethod
    def default() -> TokenBudget:
        """Create a sensible default budget for coding tasks."""
        return TokenBudget(
            total_budget=80_000,
            layer_percentages={
                "system": 10,          # System prompt, global instructions
                "task": 15,             # Current task description
                "working_memory": 10,   # In-flight state
                "repository": 30,       # Relevant files and symbols
                "tool_results": 15,     # Recent tool outputs
                "failure_memory": 10,   # Past errors
                "conversation": 5,      # Compressed history
                "constraint": 5,        # Hard constraints
            },
        )

    @staticmethod
    def planning() -> TokenBudget:
        """Budget optimized for the planning phase."""
        return TokenBudget(
            total_budget=60_000,
            layer_percentages={
                "system": 5,
                "task": 10,
                "repository": 50,       # Need broad repo overview
                "symbol": 20,
                "constraint": 10,
                "conversation": 5,
            },
        )

    @staticmethod
    def coding() -> TokenBudget:
        """Budget optimized for the coding phase."""
        return TokenBudget(
            total_budget=80_000,
            layer_percentages={
                "system": 10,
                "task": 15,
                "working_memory": 10,
                "repository": 40,       # Need focused file context
                "symbol": 15,
                "tool_results": 5,
                "failure_memory": 5,
            },
        )

    @staticmethod
    def debugging() -> TokenBudget:
        """Budget optimized for debugging."""
        return TokenBudget(
            total_budget=80_000,
            layer_percentages={
                "system": 5,
                "task": 10,
                "working_memory": 10,
                "repository": 15,       # Focused on failing files
                "failure_memory": 35,   # Errors and traces are critical
                "tool_results": 20,     # Test outputs, error logs
                "constraint": 5,
            },
        )


# ---------------------------------------------------------------------------
# ContextBundle ? the assembled output of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class ContextBundle:
    """The final assembled context ready for prompt injection.

    Contains:
    - Organized fragments by layer
    - Token usage statistics
    - Assembly metadata
    """

    fragments: list[ContextFragment] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    total_tokens_used: int = 0
    budget: TokenBudget = field(default_factory=TokenBudget.default)
    compression_ratio: float = 0.0
    assembly_time_ms: float = 0.0
    discarded_fragments: list[ContextFragment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_layer(self, source: ContextSource) -> list[ContextFragment]:
        """Get all fragments from a specific source layer."""
        return [f for f in self.fragments if f.source == source]

    @property
    def layer_counts(self) -> dict[str, int]:
        """Count fragments per source layer."""
        counts: dict[str, int] = {}
        for f in self.fragments:
            key = f.source.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Context retrieval request ? what the orchestrator is asked to produce
# ---------------------------------------------------------------------------

@dataclass
class ContextRequest:
    """Specification for a context build operation.

    Describes WHAT context is needed, not HOW to get it.
    The ContextOrchestrator interprets this and runs the pipeline.
    """

    task_id: str = ""
    task_title: str = ""
    task_description: str = ""
    goal: str = ""
    execution_state: ExecutionState = ExecutionState.CODING
    token_budget: TokenBudget | None = None
    working_dir: str = "."

    # Hints for retrieval
    focus_files: list[str] = field(default_factory=list)
    focus_symbols: list[str] = field(default_factory=list)
    recent_errors: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    # Graph state for dependency-aware retrieval
    dependency_ids: list[str] = field(default_factory=list)
    completed_artifact_map: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Extra metadata (e.g., downstream task titles for contract generation)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_context: Any | None = None

    # Constraints
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
