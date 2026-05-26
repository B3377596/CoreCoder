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
    ExecutionPolicyContextLayer,
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

    Splits the assembled context into two messages:
    - user_message: Goal + Current Task (the actual instruction)
    - context_message: Working Memory, Constraints, Repo files, Symbols,
      Dependencies, Failures, Artifacts — injected as an *assistant* message
      so structured environment metadata doesn't pollute the user instruction.
    """

    bundle: ContextBundle
    user_message: str = ""
    context_message: str = ""
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
        # result.user_message is Goal + Current Task
        # result.context_message is everything else (assistant-role)
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
        self._policy_layer = ExecutionPolicyContextLayer()

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

        # NOTE: System layer is intentionally SKIPPED for task execution.
        # The system prompt is already injected by the Agent as a system
        # role message (via _full_messages()).  Putting it in the user
        # message body wastes tokens and confuses the model.
        # System layer fragments are excluded here.

        # Task layer
        fragments.extend(self._task_layer.produce(request))

        # Working memory layer
        fragments.extend(self._memory_layer.produce(request))

        # Failure memory layer
        fragments.extend(self._failure_layer.produce(request))

        # Constraint layer
        fragments.extend(self._constraint_layer.produce(request))

        # Execution policy layer (task contract: bounds, stop conditions)
        fragments.extend(self._policy_layer.produce(request))

        # Repository layer (graph-aware retrieval)
        repo_options = get_retrieval_options(request.execution_state)
        repo_fragments = self._retriever.retrieve(request, repo_options)
        fragments.extend(repo_fragments)

        # Empty project detection: if no repo content was found, explicitly
        # tell the agent this is a blank slate so it doesn't waste rounds
        # exploring.
        if not repo_fragments and not request.completed_artifact_map:
            fragments.append(ContextFragment(
                source=ContextSource.SYSTEM,
                type=ContextType.INSTRUCTION,
                content=(
                    "## Project State\n"
                    "EMPTY PROJECT — no code, no venv, no config, no package manager. "
                    "Start everything from scratch. "
                    "Do NOT explore or list files to confirm — there is nothing here."
                ),
                priority=10,
                relevance_score=1.0,
                token_count=40,
            ))

        # ---- Phase 2: Pipeline (rank → deduplicate → compress → budget) ----
        bundle = self._pipeline.run(fragments, request, budget)

        # ---- Phase 3: Assemble messages ----
        user_message = self._assemble_user_message(bundle)
        context_message = self._assemble_context_message(bundle)
        print(user_message)
        print("-" * 40)
        print(context_message)
        assembly_time_ms = (time.time() - t0) * 1000.0

        # Collect fragment counts by source
        fragment_counts: dict[str, int] = {}
        for f in bundle.fragments:
            key = f.source.value
            fragment_counts[key] = fragment_counts.get(key, 0) + 1

        return AssemblyResult(
            bundle=bundle,
            user_message=user_message,
            context_message=context_message,
            assembly_time_ms=assembly_time_ms,
            fragment_counts=fragment_counts,
        )

    # ------------------------------------------------------------------
    # prompt assembly
    # ------------------------------------------------------------------

    def _assemble_user_message(self, bundle: ContextBundle) -> str:
        """Assemble the USER message — only Goal + Current Task.

        These are the actual instructions the agent must act on.
        Everything else (working memory, repo files, constraints) goes
        into the context (assistant) message.
        """
        parts: list[str] = []
        for frag in bundle.fragments:
            if frag.source == ContextSource.TASK:
                parts.append(frag.content)
                parts.append("")
        return "\n".join(parts).strip()

    def _assemble_context_message(self, bundle: ContextBundle) -> str:
        """Assemble the CONTEXT (assistant) message — structured environment info.

        Includes: Working Memory, Repo Files, Symbols, Dependencies,
        Completed Artifacts, Failures, Constraints, Tool Results.
        Excludes TASK source (that goes in the user message).
        """
        sections: list[tuple[ContextSource, str]] = [
            (ContextSource.WORKING_MEMORY, "Working Memory"),
            (ContextSource.CONSTRAINT, "Constraints"),
            (ContextSource.REPOSITORY, "Repository"),
            (ContextSource.SYMBOL, "Symbols"),
            (ContextSource.DEPENDENCY_GRAPH, "Dependencies"),
            (ContextSource.ARTIFACT, "Completed Tasks"),
            (ContextSource.FAILURE_MEMORY, "Failures"),
            (ContextSource.TOOL_RESULT, "Tool Results"),
        ]

        parts: list[str] = []
        for source, _ in sections:
            layer_fragments = [
                f for f in bundle.fragments if f.source == source
            ]
            if not layer_fragments:
                continue
            for frag in layer_fragments:
                parts.append(frag.content)
                parts.append("")

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
        downstream_tasks: list[str] | None = None,
        token_budget: TokenBudget | None = None,
    ) -> AssemblyResult:
        """Convenience method — builds the ContextRequest and calls build_context()."""
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
            metadata={"downstream_tasks": downstream_tasks or []},
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
