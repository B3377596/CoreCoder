"""Central Context Orchestrator — the runtime context assembly engine.

Sits BETWEEN the Scheduler and Executor in the DAG pipeline:

    Scheduler → ContextOrchestrator.build_context() → Executor → Agent

The orchestrator is the single entry point for context assembly.  It:
1. Receives a ContextRequest from the scheduler
2. Invokes each context layer to collect candidate fragments
3. Retrieves repository context via the graph-aware retriever
4. Runs the assembly pipeline (rank → deduplicate → compress → budget)
5. Assembles the final prompt
6. Returns a fully baked ContextBundle + formatted prompt string

Design invariants:
- The orchestrator owns NO LLM state.  It's a pure context engine.
- All retrieval is lazy (cached per-run, invalidated on state change).
- The output is always a typed ContextBundle, never a raw string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextBundle,
    ContextRequest,
    TokenBudget,
    ExecutionState,
    ContextSource,
    ContextType,
)
from corecoder.orchestration.context.layers import (
    SystemContextLayer,
    TaskContextLayer,
    WorkingMemoryContextLayer,
    FailureMemoryContextLayer,
    ConstraintContextLayer,
)
from corecoder.orchestration.context.ranker import ContextRanker
from corecoder.orchestration.context.retriever import RepositoryContextRetriever
from corecoder.orchestration.context.pipeline import ContextAssemblyPipeline
from corecoder.orchestration.context.policies import (
    get_policy,
    get_retrieval_options,
    StatePolicy,
)


@dataclass
class AssemblyResult:
    """Output of the ContextOrchestrator.build_context() call.

    Contains both the structured bundle and the formatted prompt ready
    for injection into the agent.
    """

    bundle: ContextBundle
    prompt: str = ""
    assembly_time_ms: float = 0.0
    fragment_counts: dict[str, int] = field(default_factory=dict)


class ContextOrchestrator:
    """Central context orchestration engine.

    Usage:
        orch = ContextOrchestrator(working_dir="/path/to/repo")
        orch.set_system_prompt(system_prompt)

        request = ContextRequest(
            task_title="Implement auth",
            goal="Build login system",
            execution_state=ExecutionState.CODING,
            focus_files=["auth.py"],
        )
        result = orch.build_context(request)
        # result.prompt is ready for Agent.chat()
    """

    def __init__(
        self,
        working_dir: str = ".",
        config: ContextOrchestratorConfig | None = None,
    ):
        self._config = config or ContextOrchestratorConfig()
        self._working_dir = working_dir

        # Layers
        self._system_layer = SystemContextLayer()
        self._task_layer = TaskContextLayer()
        self._memory_layer = WorkingMemoryContextLayer()
        self._failure_layer = FailureMemoryContextLayer()
        self._constraint_layer = ConstraintContextLayer()

        # Retrieval
        self._retriever = RepositoryContextRetriever(working_dir)

        # Pipeline
        self._ranker = ContextRanker()
        self._pipeline = ContextAssemblyPipeline(ranker=self._ranker)

        # Cache (per-run — cleared on each build_context call by default)
        self._fragment_cache: dict[str, list[ContextFragment]] = {}

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        self._system_layer.set_system_prompt(prompt)

    def set_repo_retriever(self, retriever: RepositoryContextRetriever) -> None:
        self._retriever = retriever

    def set_ranker(self, ranker: ContextRanker) -> None:
        self._ranker = ranker
        self._pipeline = ContextAssemblyPipeline(ranker=ranker)

    def add_pipeline_stage(self, name: str, stage_fn, after: str = "") -> None:
        self._pipeline.add_stage(name, stage_fn, after)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def build_context(self, request: ContextRequest) -> AssemblyResult:
        """Build execution context for a task.

        This is the single method the scheduler calls before each task
        execution.  Returns a fully assembled prompt string plus the
        structured bundle for observability.

        Args:
            request: What context is needed (task, state, files, etc.)

        Returns:
            AssemblyResult with prompt string and structured bundle.
        """
        t0 = time.time()

        # Resolve execution state policy
        policy = get_policy(request.execution_state)
        budget = request.token_budget or policy.token_budget

        # ---- Phase 1: Collect candidates ----
        fragments: list[ContextFragment] = []

        # System layer
        fragments.extend(self._system_layer.produce(request))

        # Task layer
        fragments.extend(self._task_layer.produce(request))

        # Working memory layer
        fragments.extend(self._memory_layer.produce(request))

        # Failure memory layer
        fragments.extend(self._failure_layer.produce(request))

        # Constraint layer
        fragments.extend(self._constraint_layer.produce(request))

        # Repository layer (graph-aware retrieval)
        repo_options = get_retrieval_options(request.execution_state)
        repo_fragments = self._retriever.retrieve(request, repo_options)
        fragments.extend(repo_fragments)

        # ---- Phase 2: Pipeline (rank → deduplicate → compress → budget) ----
        bundle = self._pipeline.run(fragments, request, budget)

        # ---- Phase 3: Assemble prompt ----
        prompt = self._assemble_prompt(bundle, request)

        assembly_time_ms = (time.time() - t0) * 1000.0

        # Collect fragment counts by source
        fragment_counts: dict[str, int] = {}
        for f in bundle.fragments:
            key = f.source.value
            fragment_counts[key] = fragment_counts.get(key, 0) + 1

        return AssemblyResult(
            bundle=bundle,
            prompt=prompt,
            assembly_time_ms=assembly_time_ms,
            fragment_counts=fragment_counts,
        )

    # ------------------------------------------------------------------
    # prompt assembly
    # ------------------------------------------------------------------

    def _assemble_prompt(
        self, bundle: ContextBundle, request: ContextRequest
    ) -> str:
        """Assemble the final prompt from the context bundle.

        The prompt structure follows a consistent format:
        1. System context (always first)
        2. Task context (goal + current task)
        3. Repository context (relevant files, symbols)
        4. Working memory (assumptions, completed work)
        5. Failure memory (if any)
        6. Constraints (always near the end)

        Fragments are already sorted by relevance score from the pipeline.
        """
        sections: list[tuple[str, ContextSource, str]] = [
            ("System", ContextSource.SYSTEM, ""),
            ("Task", ContextSource.TASK, ""),
            ("Repository", ContextSource.REPOSITORY, ""),
            ("Symbols", ContextSource.SYMBOL, ""),
            ("Dependencies", ContextSource.DEPENDENCY_GRAPH, ""),
            ("Working Memory", ContextSource.WORKING_MEMORY, ""),
            ("Artifacts", ContextSource.ARTIFACT, ""),
            ("Tool Results", ContextSource.TOOL_RESULT, ""),
            ("Failures", ContextSource.FAILURE_MEMORY, ""),
            ("Constraints", ContextSource.CONSTRAINT, ""),
        ]

        parts: list[str] = []

        for section_name, source, _ in sections:
            layer_fragments = [
                f for f in bundle.fragments if f.source == source
            ]
            if not layer_fragments:
                continue

            for frag in layer_fragments:
                parts.append(frag.content)
                parts.append("")  # Blank line between fragments

        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # convenience: direct build for the executor integration point
    # ------------------------------------------------------------------

    def build_task_context(
        self,
        task_id: str,
        task_title: str,
        task_description: str,
        goal: str,
        execution_state: ExecutionState = ExecutionState.CODING,
        focus_files: list[str] | None = None,
        focus_symbols: list[str] | None = None,
        recent_errors: list[str] | None = None,
        constraints: list[str] | None = None,
        assumptions: list[str] | None = None,
        dependency_ids: list[str] | None = None,
        completed_artifacts: dict[str, dict[str, Any]] | None = None,
        token_budget: TokenBudget | None = None,
    ) -> AssemblyResult:
        """Convenience method — builds the ContextRequest and calls build_context().

        This is the integration point for the Executor/Scheduler.
        """
        request = ContextRequest(
            task_id=task_id,
            task_title=task_title,
            task_description=task_description,
            goal=goal,
            execution_state=execution_state,
            token_budget=token_budget,
            working_dir=self._working_dir,
            focus_files=focus_files or [],
            focus_symbols=focus_symbols or [],
            recent_errors=recent_errors or [],
            constraints=constraints or [],
            assumptions=assumptions or [],
            dependency_ids=dependency_ids or [],
            completed_artifact_map=completed_artifacts or {},
        )
        return self.build_context(request)

    def clear_cache(self) -> None:
        """Clear the fragment cache."""
        self._fragment_cache.clear()
        self._retriever.invalidate_cache()


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ContextOrchestratorConfig:
    """Configuration for the ContextOrchestrator."""

    # Default budget
    default_budget: TokenBudget | None = None

    # Retrieval defaults
    default_max_files: int = 10
    default_max_symbols: int = 20
    default_dependency_radius: int = 2

    # Compression defaults
    compress_aggressively: bool = False
    max_lines_per_file: int = 200

    # Observability
    log_assembly: bool = True
    log_discarded: bool = False
    log_token_usage: bool = True
