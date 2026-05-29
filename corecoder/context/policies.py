"""Execution-state-based context policies.

Different execution phases require radically different context profiles.
A coding agent doesn't need the full repository overview, and a debugging
agent doesn't need the task graph.

This module defines which context layers are active, their token budgets,
and retrieval behaviors for each execution state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from corecoder.context.models import (
    ExecutionState,
    TokenBudget,
)
from corecoder.context.retriever import RetrievalOptions


@dataclass
class StatePolicy:
    """Context policy for a specific execution state.

    Defines:
    - Active context layers and their token budgets
    - Retrieval options (depth, breadth)
    - Compression aggressiveness
    - Anti-noise settings
    """

    state: ExecutionState
    description: str = ""
    token_budget: TokenBudget = field(default_factory=TokenBudget.default)

    # Retrieval tuning
    max_files: int = 10
    max_symbols: int = 20
    dependency_radius: int = 2
    include_callers: bool = True
    include_callees: bool = True

    # Compression tuning
    compress_aggressively: bool = False
    max_lines_per_file: int = 200

    # Anti-noise
    min_fragment_length: int = 10
    max_fragments_per_layer: int = 50


# ===========================================================================
# State-specific policies
# ===========================================================================

POLICIES: dict[ExecutionState, StatePolicy] = {
    ExecutionState.PLANNING: StatePolicy(
        state=ExecutionState.PLANNING,
        description="Decomposing goal into task graph ?*broad overview needed",
        token_budget=TokenBudget.planning(),
        max_files=20,           # Broad overview
        max_symbols=30,
        dependency_radius=3,    # Deep exploration
        include_callers=False,
        include_callees=True,
        max_lines_per_file=100,  # Shorter previews
    ),

    ExecutionState.EXPLORING: StatePolicy(
        state=ExecutionState.EXPLORING,
        description="Understanding codebase structure",
        token_budget=TokenBudget(
            total_budget=80_000,
            layer_percentages={
                "system": 5,
                "task": 10,
                "repository": 50,
                "symbol": 20,
                "dependency_graph": 10,
                "constraint": 5,
            },
        ),
        max_files=15,
        max_symbols=25,
        dependency_radius=3,
    ),

    ExecutionState.CODING: StatePolicy(
        state=ExecutionState.CODING,
        description="Writing/editing code ?*focused file context",
        token_budget=TokenBudget.coding(),
        max_files=8,            # Focused, not broad
        max_symbols=15,
        dependency_radius=1,    # Immediate neighbors only
        include_callers=True,
        include_callees=True,
        max_lines_per_file=300,
    ),

    ExecutionState.TESTING: StatePolicy(
        state=ExecutionState.TESTING,
        description="Running tests and validating",
        token_budget=TokenBudget(
            total_budget=60_000,
            layer_percentages={
                "system": 5,
                "task": 10,
                "working_memory": 10,
                "tool_result": 40,      # Test output is critical
                "failure_memory": 25,
                "repository": 10,
            },
        ),
        max_files=5,            # Only the files being tested
        max_symbols=10,
        dependency_radius=1,
    ),

    ExecutionState.DEBUGGING: StatePolicy(
        state=ExecutionState.DEBUGGING,
        description="Investigating failures ?*errors and traces are critical",
        token_budget=TokenBudget.debugging(),
        max_files=6,            # Focused on failing files
        max_symbols=10,
        dependency_radius=2,
        include_callers=True,   # Who called the failing code
        include_callees=False,
        max_lines_per_file=400,  # Need full context for debugging
    ),

    ExecutionState.VERIFYING: StatePolicy(
        state=ExecutionState.VERIFYING,
        description="Checking completion criteria",
        token_budget=TokenBudget(
            total_budget=40_000,
            layer_percentages={
                "system": 5,
                "task": 15,
                "tool_result": 50,
                "failure_memory": 20,
                "constraint": 10,
            },
        ),
        max_files=3,
        max_symbols=5,
        dependency_radius=0,    # No exploration needed
    ),

    ExecutionState.REFACTORING: StatePolicy(
        state=ExecutionState.REFACTORING,
        description="Restructuring existing code",
        token_budget=TokenBudget.coding(),
        max_files=10,
        max_symbols=20,
        dependency_radius=2,    # Need to see callers
        include_callers=True,
        include_callees=True,
        max_lines_per_file=300,
    ),
}


def get_policy(state: ExecutionState) -> StatePolicy:
    """Get the context policy for a given execution state."""
    return POLICIES.get(state, POLICIES[ExecutionState.CODING])


def get_retrieval_options(state: ExecutionState) -> RetrievalOptions:
    """Extract RetrievalOptions from the state policy."""
    policy = get_policy(state)
    return RetrievalOptions(
        max_files=policy.max_files,
        max_symbols=policy.max_symbols,
        dependency_radius=policy.dependency_radius,
        include_callers=policy.include_callers,
        include_callees=policy.include_callees,
    )
