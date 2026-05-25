"""Planner layer — converts user goals into executable TaskGraphs.

The planner is the "brain" of the orchestration system.  It takes a
high-level user goal and produces a structured DAG of tasks with
dependencies.  The execution layer then follows this plan.

Planner interface is deliberately abstract to support multiple backends:
- StaticPlanner: predefined task templates (testing, demos)
- LLMPlanner: uses an LLM to decompose goals (production path)
- Future: ReplayPlanner (replay a saved plan), HybridPlanner, etc.

Design contract: planners produce a TaskGraph.  They do NOT execute anything.
This separation means we can test planning logic without an agent, and test
execution logic with hand-crafted graphs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from corecoder.orchestration.dag.models import TaskNode, TaskStatus, RetryPolicy
from corecoder.orchestration.dag.graph import TaskGraph


@dataclass
class PlanResult:
    """Output of a planning invocation."""

    graph: TaskGraph
    plan_summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    estimated_tasks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePlanner(ABC):
    """Abstract planner — converts a goal into a TaskGraph.

    Implementations:
    - StaticPlanner: manual task definition
    - LLMPlanner: LLM-driven decomposition
    - (future) ReplayPlanner: replay a saved plan
    - (future) InteractivePlanner: user-guided planning
    """

    @abstractmethod
    def plan(self, goal: str, context: dict[str, Any] | None = None) -> PlanResult:
        """Produce a task graph from a user goal.

        Args:
            goal: The user's high-level objective.
            context: Optional context dict with keys like:
                - working_dir: str
                - repo_summary: str
                - existing_files: list[str]
                - previous_plan: dict (for replanning)

        Returns:
            PlanResult with the task graph and metadata.
        """

    def replan(
        self,
        original_plan: PlanResult,
        failure_context: dict[str, Any],
    ) -> PlanResult:
        """Modify a plan in response to failures.

        Default implementation: returns the original plan unchanged.
        Subclasses override this to insert recovery tasks.
        """
        return original_plan


class StaticPlanner(BasePlanner):
    """Planner that builds a graph from a predefined recipe.

    The recipe is a list of task definitions with dependencies specified
    by index (0-based) into the list.  This is useful for:
    - Testing the orchestration layer without an LLM
    - Demo scripts
    - Template-based planning where tasks are known in advance

    Recipe format:
        [
            {"title": "Set up project", "description": "...", "priority": 10},
            {"title": "Write tests", "description": "...", "deps": [0], "priority": 5},
            ...
        ]
    """

    def __init__(self, recipe: list[dict[str, Any]] | None = None):
        self._recipe = recipe or []

    def set_recipe(self, recipe: list[dict[str, Any]]) -> None:
        self._recipe = recipe

    def plan(self, goal: str, context: dict[str, Any] | None = None) -> PlanResult:
        graph = TaskGraph(name="static_plan")
        nodes: list[TaskNode] = []

        # First pass: create all nodes
        for i, task_def in enumerate(self._recipe):
            node = TaskNode(
                title=task_def.get("title", f"Task {i}"),
                description=task_def.get("description", ""),
                priority=task_def.get("priority", 0),
                retry_policy=RetryPolicy(
                    max_retries=task_def.get("max_retries", 3)
                ),
                metadata=task_def.get("metadata", {}),
            )
            graph.add_node(node)
            nodes.append(node)

        # Second pass: wire dependencies
        for i, task_def in enumerate(self._recipe):
            for dep_idx in task_def.get("deps", []):
                if 0 <= dep_idx < len(nodes) and dep_idx != i:
                    graph.add_dependency(nodes[i].id, nodes[dep_idx].id)

        return PlanResult(
            graph=graph,
            plan_summary=f"Static plan: {goal}",
            estimated_tasks=len(nodes),
        )


class LLMPlanner(BasePlanner):
    """LLM-driven planner that decomposes a user goal into a task graph.

    Uses an LLM to:
    1. Analyze the goal
    2. Break it into subtasks
    3. Identify dependencies between subtasks
    4. Assign priorities

    The LLM is expected to produce a JSON structure that this planner
    parses into a TaskGraph.  This is a placeholder — the actual
    implementation depends on the specific LLM interface being used.

    Design note: the LLMPlanner does NOT own an LLM instance.  It receives
    an `llm_call` function via dependency injection.  This keeps the planner
    decoupled from any specific LLM implementation and makes it testable
    with mock LLM responses.
    """

    # Prompt template for the planning LLM call.
    #
    # Design notes on what makes a good plan:
    # - Maximize parallelism: tasks that don't share outputs/files should be independent.
    # - Real dependency = must read the output of another task.  "Happens after" is not a dependency.
    # - Shallow and wide beats deep and narrow.  Aim for depth ≤ 3.
    # - Each task should produce concrete, verifiable artifacts (files, test results).
    PLANNING_PROMPT = """You are a software engineering planner. Given a user goal, decompose it into a
task DAG (directed acyclic graph) that a coding agent can execute.

## Goal
{goal}

## Context
{context}

## Core Principle: MAXIMIZE PARALLELISM

A dependency exists ONLY when task B genuinely cannot start until task A produces
a concrete output (file, library, config, etc.) that B needs to read or use.

"Task B happens after task A" is NOT a dependency if B doesn't consume A's output.
If two tasks touch different files, they should be INDEPENDENT — they will run
concurrently, speeding up the entire plan.

## Rules

1. Produce 3-8 tasks.  Each must be concrete: a coding agent with file I/O,
   shell, and code search tools should be able to complete it in one shot.

2. **Depth limit**: no dependency chain longer than 3 levels.
   Bad:  A → B → C → D → E → F   (depth 5, everything blocked on one path)
   Good: A → B → C,  A → D → E    (depth 2, two parallel branches)

3. **Branch at every opportunity**.  If you have N tasks that don't share
   outputs, make them all direct children of the same parent so they run
   in parallel.

4. **"Leaf" tasks (tests, lint, docs) should be terminal nodes** that depend
   on the implementation tasks they verify.  They should NOT form long chains.

5. **Priorities**: 10 = foundation/setup (must do first), 8 = core logic,
   5 = features/enhancements, 3 = tests/verification.
   Higher priority tasks are scheduled first among ready tasks.

## Thinking Step

Before writing the JSON, briefly list:
- What are the concrete outputs (files) of each task?
- Which outputs does each downstream task ACTUALLY need?
- Can any tasks run in parallel?  If yes, make them siblings under the same parent.

## Output JSON

Return ONLY a JSON object.  Tasks are ordered in the array by index (0, 1, 2...).
Dependencies reference PREVIOUS indices.  Task 0 cannot have dependencies.

{{
    "plan_summary": "One-line summary",
    "assumptions": ["..."],
    "constraints": ["..."],
    "tasks": [
        {{
            "title": "Short title",
            "description": "What this task does and what files it produces.",
            "dependencies": [],        // indices of prerequisite tasks.  EMPTY for root tasks.
            "priority": 10,
            "verification": {{
                "expected_files": ["path/to/output.py"],
                "required_patterns": ["SUCCESS", "completed", "created"],
                "forbidden_patterns": ["ERROR", "FAILED"]
            }}
        }}
    ]
}}

## Concrete Example

Goal: "Build a REST API for a todo app"

Thinking:
- Setup is the root: creates files that everything else needs
- Models and routes are independent (different files) → siblings under setup
- Tests for models depend on models, tests for routes depend on routes → leaf nodes
- Depth = 2, width = 2 parallel branches

Output:
{{
    "plan_summary": "FastAPI todo API with models, routes, and tests",
    "assumptions": ["Python 3.10+", "FastAPI available"],
    "constraints": ["In-memory storage, no database"],
    "tasks": [
        {{
            "title": "Initialize project structure",
            "description": "Create main.py, requirements.txt, and folder layout.",
            "dependencies": [],
            "priority": 10,
            "verification": {{"expected_files": ["main.py", "requirements.txt"]}}
        }},
        {{
            "title": "Implement Todo data model",
            "description": "Define Pydantic Todo model with id/title/done fields in models.py.",
            "dependencies": [0],
            "priority": 8,
            "verification": {{"expected_files": ["models.py"], "required_patterns": ["class Todo"]}}
        }},
        {{
            "title": "Implement CRUD API routes",
            "description": "Create GET/POST/PUT/DELETE /todos endpoints in routes.py.",
            "dependencies": [0],
            "priority": 8,
            "verification": {{"expected_files": ["routes.py"], "required_patterns": ["@app"]}}
        }},
        {{
            "title": "Write model unit tests",
            "description": "Test Pydantic model validation and serialization.",
            "dependencies": [1],
            "priority": 3,
            "verification": {{"expected_files": ["tests/test_models.py"], "required_patterns": ["def test_"]}}
        }},
        {{
            "title": "Write API integration tests",
            "description": "Test all endpoints with FastAPI TestClient.",
            "dependencies": [2],
            "priority": 3,
            "verification": {{"test_command": "pytest tests/test_routes.py -x", "expected_files": ["tests/test_routes.py"]}}
        }}
    ]
}}

Graph structure of this example (notice the branching):
    Task 0 (Setup) → Task 1 (Models) → Task 3 (Model tests)
                   → Task 2 (Routes) → Task 4 (Route tests)
    Depth = 2.  Models and Routes run in parallel.  Tests run in parallel.

Return ONLY the JSON.  Do NOT include the thinking step in your output — use it
internally to design the graph, then output only JSON."""

    def __init__(
        self,
        llm_call: Callable[..., Any] | None = None,
        model: str = "",
    ):
        """Initialize the LLM planner.

        Args:
            llm_call: An async callable that takes messages and returns
                      an LLM response with a .content attribute.  This is
                      typically an Agent or LLM instance's chat method.
            model: Optional model identifier for logging.
        """
        self._llm_call = llm_call
        self._model = model

    def set_llm(self, llm_call: Callable[..., Any]) -> None:
        """Inject an LLM callable after construction."""
        self._llm_call = llm_call

    async def aplan(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> PlanResult:
        """Async version: call the LLM and parse its response into a graph."""
        if self._llm_call is None:
            raise RuntimeError("LLMPlanner has no LLM callable.  Call set_llm() first.")

        ctx_str = json.dumps(context or {}, indent=2, default=str)
        prompt = self.PLANNING_PROMPT.format(goal=goal, context=ctx_str)

        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_call(messages)

        return self._parse_response(response.content, goal)

    def plan(self, goal: str, context: dict[str, Any] | None = None) -> PlanResult:
        """Synchronous plan — returns an empty graph with a hint to use aplan().

        For synchronous usage, use StaticPlanner or call aplan() directly.
        """
        graph = TaskGraph(name="llm_plan_pending")
        return PlanResult(
            graph=graph,
            plan_summary="LLM plan pending — use aplan() for async planning",
            estimated_tasks=0,
            metadata={"status": "pending", "hint": "Call aplan() instead"},
        )

    def _parse_response(self, response_text: str, goal: str) -> PlanResult:
        """Parse the LLM's JSON response into a TaskGraph.

        Handles several LLM output patterns:
        1. Pure JSON
        2. ```json ... ``` fenced JSON
        3. Thinking text followed by a JSON block
        4. Thinking text with inline JSON object
        """
        text = response_text.strip()

        # Strategy 1: Try to extract ```json ... ``` fence
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        # Strategy 2: Find the outermost { ... } JSON object
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                text = text[brace_start:brace_end + 1]

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Fallback: single task with raw goal
            graph = TaskGraph(name="llm_plan_fallback")
            node = TaskNode(
                title="Execute goal",
                description=goal,
                priority=10,
                metadata={"raw_response": response_text[:2000]},
            )
            graph.add_node(node)
            return PlanResult(
                graph=graph,
                plan_summary="Fallback: could not parse LLM plan",
                estimated_tasks=1,
                metadata={"parse_error": True},
            )

        graph = TaskGraph(name="llm_plan")
        tasks_data = data.get("tasks", [])
        nodes: list[TaskNode] = []

        # First pass: create nodes
        for i, td in enumerate(tasks_data):
            verification_meta = td.get("verification", {})
            node = TaskNode(
                title=td.get("title", f"Task {i}"),
                description=td.get("description", ""),
                priority=td.get("priority", 0),
                metadata={
                    "verification": verification_meta,
                    **{k: v for k, v in td.items() if k not in ("title", "description", "dependencies", "priority", "verification")},
                },
            )
            graph.add_node(node)
            nodes.append(node)

        # Second pass: wire dependencies
        for i, td in enumerate(tasks_data):
            for dep_idx in td.get("dependencies", []):
                if isinstance(dep_idx, int) and 0 <= dep_idx < len(nodes) and dep_idx != i:
                    try:
                        graph.add_dependency(nodes[i].id, nodes[dep_idx].id)
                    except Exception:
                        # Skip invalid dependencies rather than failing the whole plan
                        pass

        return PlanResult(
            graph=graph,
            plan_summary=data.get("plan_summary", f"Plan for: {goal}"),
            assumptions=data.get("assumptions", []),
            constraints=data.get("constraints", []),
            estimated_tasks=len(nodes),
            metadata={"raw_plan": data},
        )

    def replan(
        self,
        original_plan: PlanResult,
        failure_context: dict[str, Any],
    ) -> PlanResult:
        """Replan by inserting recovery tasks based on failure context.

        This is a synchronous heuristic replan.  For LLM-driven replanning,
        use areplan() which calls the LLM to analyze failures.
        """
        failed_task_id = failure_context.get("task_id", "")
        replan_hint = failure_context.get("replan_hint", "")
        failed_node = original_plan.graph.get_node(failed_task_id)

        if failed_node is None:
            return original_plan

        # Insert a diagnostic/investigation task before retrying
        diag_task = TaskNode(
            title=f"Diagnose: {failed_node.title}",
            description=(
                f"The task '{failed_node.title}' failed.  {replan_hint}\n\n"
                f"Investigate the failure and report findings.  "
                f"Do NOT modify code yet — just understand what went wrong."
            ),
            priority=failed_node.priority + 1,  # Higher priority than the original
            retry_policy=RetryPolicy(max_retries=1),
            metadata={"replan_task": True, "original_task": failed_task_id},
        )

        # Insert the diagnostic task before the failed task's dependents
        dependents = original_plan.graph.get_dependents(failed_task_id)
        if dependents:
            for dep_id in dependents:
                try:
                    original_plan.graph.insert_node_between(
                        failed_task_id, dep_id, diag_task
                    )
                except Exception:
                    pass
        else:
            # No dependents — just add as a dependent of the failed task
            original_plan.graph.add_node(diag_task)
            try:
                original_plan.graph.add_dependency(diag_task.id, failed_task_id)
            except Exception:
                pass

        return original_plan
