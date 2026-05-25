"""Working memory system for task orchestration.

Each task node executes within a context that includes:
- The overall user goal
- What the current task is supposed to do
- Known constraints discovered during execution
- Recent failures (so the LLM can avoid repeating them)
- Assumptions made by upstream tasks
- Artifacts produced by completed dependencies

This memory is injected into the LLM prompt before each task execution,
giving the agent grounded context without polluting the conversation
history with every detail of the orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from corecoder.orchestration.dag.models import TaskStatus


@dataclass
class WorkingMemory:
    """Per-task execution context.

    This is the "consciousness" of a single task node — everything the
    agent needs to know to execute this specific piece of work, and
    nothing it doesn't.

    The memory is built by the scheduler before handing a task to the
    executor, and is cleared after the task completes (success or failure).
    """

    current_goal: str = ""
    current_task_id: str = ""
    current_task_title: str = ""
    current_task_description: str = ""

    # Constraints discovered or declared
    known_constraints: list[str] = field(default_factory=list)

    # Failures from this task's previous attempts
    recent_failures: list[str] = field(default_factory=list)

    # Assumptions carried forward from dependency tasks
    assumptions: list[str] = field(default_factory=list)

    # Artifacts from completed upstream tasks
    # Maps task_id -> {description, files, outputs, ...}
    completed_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Downstream task titles — shown to agent so it knows what NOT to do
    downstream_tasks: list[str] = field(default_factory=list)

    # Free-form notes the agent can read/write
    notes: str = ""

    # Metadata about the overall run
    run_id: str = ""
    plan_summary: str = ""

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self.known_constraints:
            self.known_constraints.append(constraint)

    def add_failure(self, failure: str) -> None:
        self.recent_failures.append(failure)

    def add_assumption(self, assumption: str) -> None:
        if assumption not in self.assumptions:
            self.assumptions.append(assumption)

    def add_artifact(self, task_id: str, artifact: dict[str, Any]) -> None:
        self.completed_artifacts[task_id] = artifact

    def to_prompt_context(self) -> str:
        """Render working memory as a prompt section for the LLM.

        The format is deliberately terse to minimize token usage while
        preserving all actionable information.
        """
        parts: list[str] = []

        parts.append(f"## Current Goal\n{self.current_goal}")

        if self.plan_summary:
            parts.append(f"\n## Plan Overview\n{self.plan_summary}")

        parts.append(f"\n## Current Task\n**{self.current_task_title}**\n{self.current_task_description}")

        if self.known_constraints:
            parts.append("\n## Constraints")
            for c in self.known_constraints:
                parts.append(f"- {c}")

        if self.assumptions:
            parts.append("\n## Assumptions from Previous Tasks")
            for a in self.assumptions:
                parts.append(f"- {a}")

        if self.completed_artifacts:
            parts.append("\n## Completed Work (from dependencies) — already done, do NOT repeat")
            for tid, art in self.completed_artifacts.items():
                desc = art.get("description", tid)
                parts.append(f"- COMPLETED: **{desc}**")
                # Show what files were created — this is the key working memory
                created = (art.get("created_files", []) or
                          art.get("all_changed", []) or
                          art.get("agent_mentioned_files", []))
                if created:
                    parts.append(f"  Files: {', '.join(str(f) for f in created[:10])}")
                # Other metadata
                for key, value in art.items():
                    if key in ("description", "created_files", "all_changed",
                               "agent_mentioned_files", "modified_files", "deleted_files"):
                        continue
                    if isinstance(value, list):
                        parts.append(f"  {key}: {', '.join(str(v) for v in value)}")
                    elif isinstance(value, (str, int, float)):
                        parts.append(f"  {key}: {value}")

        if self.recent_failures:
            parts.append("\n## Previous Failures (do NOT repeat these approaches)")
            for f in self.recent_failures:
                parts.append(f"- {f}")

        if self.notes:
            parts.append(f"\n## Notes\n{self.notes}")

        return "\n".join(parts)

    def clear(self) -> None:
        """Reset all fields for the next task."""
        self.current_task_id = ""
        self.current_task_title = ""
        self.current_task_description = ""
        self.known_constraints.clear()
        self.recent_failures.clear()
        self.assumptions.clear()
        self.completed_artifacts.clear()
        self.notes = ""


class MemoryInjector:
    """Builds and injects working memory into task execution.

    This is a stateless factory — it reads the graph state and
    constructs a WorkingMemory for a specific task node.

    Usage:
        injector = MemoryInjector()
        memory = injector.build(task_node, graph, goal="Build a web app")
        prompt_context = memory.to_prompt_context()
    """

    def build(
        self,
        task_id: str,
        graph,  # TaskGraph (avoid circular import)
        goal: str = "",
        run_id: str = "",
        plan_summary: str = "",
        extra_constraints: list[str] | None = None,
        extra_assumptions: list[str] | None = None,
    ) -> WorkingMemory:
        """Construct working memory for a task by reading graph state."""
        node = graph.get_node(task_id)
        if node is None:
            raise KeyError(f"Task not found: {task_id}")

        memory = WorkingMemory(
            current_goal=goal,
            current_task_id=node.id,
            current_task_title=node.title,
            current_task_description=node.description,
            known_constraints=list(extra_constraints or []),
            assumptions=list(extra_assumptions or []),
            run_id=run_id,
            plan_summary=plan_summary,
        )

        # Collect artifacts from completed dependencies
        for dep_id in node.dependencies:
            dep_node = graph.get_node(dep_id)
            if dep_node is None:
                continue
            if dep_node.status in ("success", TaskStatus.SUCCESS) and dep_node.artifacts:
                memory.add_artifact(dep_id, {
                    "description": dep_node.title,
                    **dep_node.artifacts,
                })

        # Carry forward assumptions from dependencies
        for dep_id in node.dependencies:
            dep_node = graph.get_node(dep_id)
            if dep_node is None:
                continue
            for assumption in dep_node.metadata.get("assumptions", []):
                memory.add_assumption(assumption)

        # Collect constraints from dependency metadata
        for dep_id in node.dependencies:
            dep_node = graph.get_node(dep_id)
            if dep_node is None:
                continue
            for constraint in dep_node.metadata.get("constraints", []):
                memory.add_constraint(constraint)

        # Add recent failures from this task's previous attempts
        if node.result and not node.result.success:
            memory.add_failure(
                f"Attempt {node.retry_count}: {node.result.error or 'Unknown error'}"
            )
        for prev_error in node.metadata.get("failure_history", []):
            memory.add_failure(prev_error)

        return memory
