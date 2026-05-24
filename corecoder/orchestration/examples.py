"""Examples demonstrating the orchestration layer.

This file contains runnable examples covering:
1. Basic DAG construction and cycle detection
2. Topological sort
3. StaticPlanner with a recipe
4. Scheduler with mock executor
5. Recovery and retry
6. Verification pipeline
7. Working memory injection
8. Persistence (save/load/resume)
9. Dynamic replanning (insert_node_between)
10. Full Orchestrator pipeline

All examples use a mock agent callable so they run without an LLM.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

# ---------------------------------------------------------------------------
# Example 1: Basic DAG — build a graph, detect cycles
# ---------------------------------------------------------------------------

def example_1_dag_basics():
    """Build a 4-node DAG, verify cycle detection, topological sort."""
    from corecoder.orchestration.models import TaskNode
    from corecoder.orchestration.graph import TaskGraph, CycleDetectedError

    print("=" * 60)
    print("EXAMPLE 1: DAG Basics")
    print("=" * 60)

    graph = TaskGraph(name="example_1")

    # Create nodes
    a = TaskNode(id="A", title="Setup project", description="Initialize the repo")
    b = TaskNode(id="B", title="Write core module", description="Implement business logic")
    c = TaskNode(id="C", title="Write tests", description="Unit tests for core module")
    d = TaskNode(id="D", title="Run CI", description="Execute CI pipeline")

    for node in [a, b, c, d]:
        graph.add_node(node)

    # A → B → C → D
    graph.add_dependency("B", "A")
    graph.add_dependency("C", "B")
    graph.add_dependency("D", "C")

    print(f"Graph: {graph}")
    print(f"Topological order: {[n.id for n in graph.topological_sort()]}")
    print(f"Cycles: {graph.detect_cycles()}")

    # Try to add a cycle (D → A would create A→B→C→D→A)
    try:
        graph.add_dependency("A", "D")
        print("ERROR: Should have detected cycle!")
    except CycleDetectedError as e:
        print(f"Cycle correctly rejected: {e}")

    # Mark tasks as they complete
    graph.mark_completed("A")
    print(f"Ready after A: {[n.id for n in graph.get_ready_tasks()]}")
    graph.mark_completed("B")
    graph.mark_completed("C")
    graph.mark_completed("D")
    print(f"All complete? {graph.is_complete()}, All success? {graph.is_successful()}")
    print(f"Progress: {graph.progress()}")


# ---------------------------------------------------------------------------
# Example 2: Failure propagation
# ---------------------------------------------------------------------------

def example_2_failure_propagation():
    """Demonstrate how failure propagates to downstream tasks."""
    from corecoder.orchestration.models import TaskNode, TaskStatus
    from corecoder.orchestration.graph import TaskGraph

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Failure Propagation")
    print("=" * 60)

    graph = TaskGraph(name="example_2")

    a = TaskNode(id="A", title="DB Migration", description="Run migration")
    b = TaskNode(id="B", title="API Endpoint", description="Create endpoint")
    c = TaskNode(id="C", title="Frontend", description="Build UI")
    d = TaskNode(id="D", title="Deploy", description="Ship to prod")

    for node in [a, b, c, d]:
        graph.add_node(node)

    # A → B → D  and  A → C → D
    graph.add_dependency("B", "A")
    graph.add_dependency("C", "A")
    graph.add_dependency("D", "B")
    graph.add_dependency("D", "C")

    # Simulate: A succeeds, B fails
    graph.mark_completed("A")
    graph.mark_failed("B", "DB connection timeout")

    print(f"After B fails: B={graph.get_node('B').status.value}")
    print(f"  C status: {graph.get_node('C').status.value}")
    print(f"  D status: {graph.get_node('D').status.value}")
    print(f"  Blocked tasks: {[n.id for n in graph.get_blocked_tasks()]}")

    # C can still run (depends only on A, which succeeded)
    ready = graph.get_ready_tasks()
    print(f"  Ready tasks: {[n.id for n in ready]}")


# ---------------------------------------------------------------------------
# Example 3: StaticPlanner with recipe
# ---------------------------------------------------------------------------

def example_3_static_planner():
    """Define a plan via recipe and build the graph."""
    from corecoder.orchestration.planner import StaticPlanner

    print("\n" + "=" * 60)
    print("EXAMPLE 3: StaticPlanner Recipe")
    print("=" * 60)

    recipe = [
        {
            "title": "Initialize project structure",
            "description": "Create src/, tests/, pyproject.toml",
            "priority": 10,
        },
        {
            "title": "Implement data models",
            "description": "Define SQLAlchemy models for User, Post, Comment",
            "deps": [0],
            "priority": 8,
        },
        {
            "title": "Implement API routes",
            "description": "FastAPI endpoints for CRUD operations",
            "deps": [1],
            "priority": 7,
        },
        {
            "title": "Write unit tests",
            "description": "pytest tests for models and routes",
            "deps": [2],
            "priority": 5,
            "verification": {
                "test_command": "pytest tests/ -x",
                "expected_files": ["tests/test_models.py", "tests/test_routes.py"],
            },
        },
        {
            "title": "Add authentication",
            "description": "JWT-based auth middleware",
            "deps": [1],
            "priority": 6,
        },
        {
            "title": "Write integration tests",
            "description": "End-to-end API tests",
            "deps": [3, 4],
            "priority": 4,
        },
    ]

    planner = StaticPlanner(recipe)
    result = planner.plan("Build a blog API")

    graph = result.graph
    print(f"Plan: {result.plan_summary}")
    print(f"Tasks: {graph.node_count}")
    print(f"Root tasks (no deps): {[n.title for n in graph.get_root_tasks()]}")
    print(f"Terminal tasks: {[n.title for n in graph.get_terminal_tasks()]}")
    print(f"Topological order:")
    for i, node in enumerate(graph.topological_sort()):
        deps = graph.get_dependencies(node.id)
        dep_titles = [graph.get_node(d).title for d in deps]
        print(f"  {i}. [{node.title}] priority={node.priority} deps={dep_titles}")


# ---------------------------------------------------------------------------
# Example 4: Full execution with mock agent
# ---------------------------------------------------------------------------

# Mock agent chat function — simulates the ReAct loop
async def _mock_agent_chat(user_input: str) -> str:
    """Simulate an agent executing a task.  The delay is proportional
    to the perceived complexity of the task."""
    # Simulate work
    await asyncio.sleep(0.1)
    return f"[DONE] Completed: {user_input[:80]}...\nFiles changed: src/models.py, tests/test_models.py"


async def _mock_agent_chat_with_failure(user_input: str) -> str:
    """Simulate an agent that sometimes fails."""
    await asyncio.sleep(0.05)
    if "fail" in user_input.lower():
        raise RuntimeError("Simulated agent failure: connection timed out")
    return f"[DONE] Completed successfully."


async def example_4_full_execution():
    """Run the full orchestration pipeline with a mock agent."""
    from corecoder.orchestration.models import TaskNode, TaskStatus
    from corecoder.orchestration.graph import TaskGraph
    from corecoder.orchestration.scheduler import Scheduler, SchedulerConfig
    from corecoder.orchestration.executor import Executor
    from corecoder.orchestration.recovery import RecoveryManager
    from corecoder.orchestration.memory import MemoryInjector
    from corecoder.orchestration.observability import OrchestrationLogger

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Full Execution with Mock Agent")
    print("=" * 60)

    # Build graph manually
    graph = TaskGraph(name="example_4")
    a = TaskNode(id="A", title="Setup project", description="Create project structure", priority=10)
    b = TaskNode(id="B", title="Write core code", description="Implement main logic", priority=5)
    c = TaskNode(id="C", title="Write tests", description="Add unit tests", priority=3)

    for node in [a, b, c]:
        graph.add_node(node)
    graph.add_dependency("B", "A")
    graph.add_dependency("C", "B")

    # Build scheduler
    executor = Executor(agent_chat_fn=_mock_agent_chat)
    recovery = RecoveryManager()
    memory_injector = MemoryInjector()
    olog = OrchestrationLogger("example_4")

    config = SchedulerConfig(
        goal="Build a simple Python module with tests",
        continue_on_failure=True,
    )

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
        recovery=recovery,
        memory_injector=memory_injector,
        olog=olog,
        config=config,
    )

    decision = await scheduler.run()

    print(f"Decision: {decision.value}")
    print(f"Progress: {graph.progress()}")
    print(f"All success? {graph.is_successful()}")
    for node in graph:
        print(f"  {node.id} [{node.title}]: {node.status.value} "
              f"(duration={node.result.duration_ms:.0f}ms)" if node.result else "")

    print(f"\nObservability summary:")
    olog_summary = olog.summary()
    print(f"  Total elapsed: {olog_summary['total_elapsed_ms']:.0f}ms")
    print(f"  Events: {olog_summary['total_events']}")
    print(f"  Transitions:")
    for t in olog_summary["transitions"]:
        print(f"    {t['task_title']}: {t['from']} → {t['to']} ({t['reason']})")


# ---------------------------------------------------------------------------
# Example 5: Retry and recovery
# ---------------------------------------------------------------------------

async def example_5_retry_and_recovery():
    """Demonstrate retry on failure, then eventual success."""
    from corecoder.orchestration.models import TaskNode, RetryPolicy
    from corecoder.orchestration.graph import TaskGraph
    from corecoder.orchestration.scheduler import Scheduler, SchedulerConfig
    from corecoder.orchestration.executor import Executor
    from corecoder.orchestration.recovery import RecoveryManager
    from corecoder.orchestration.memory import MemoryInjector
    from corecoder.orchestration.observability import OrchestrationLogger

    print("\n" + "=" * 60)
    print("EXAMPLE 5: Retry and Recovery")
    print("=" * 60)

    graph = TaskGraph(name="example_5")

    # This task will fail
    flaky = TaskNode(
        id="flaky",
        title="Flaky network operation",
        description="This task simulates a transient failure and retry",
        retry_policy=RetryPolicy(max_retries=2),
        priority=10,
    )
    graph.add_node(flaky)

    # Mock agent that fails twice then succeeds
    call_count = [0]

    async def flaky_agent(input: str) -> str:
        call_count[0] += 1
        await asyncio.sleep(0.05)
        if call_count[0] < 3:
            raise ConnectionError(f"Simulated network failure (attempt {call_count[0]})")
        return "Connected and completed successfully."

    executor = Executor(agent_chat_fn=flaky_agent)
    recovery = RecoveryManager()
    memory_injector = MemoryInjector()
    olog = OrchestrationLogger("example_5")

    config = SchedulerConfig(goal="Test retry behavior", continue_on_failure=False)

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
        recovery=recovery,
        memory_injector=memory_injector,
        olog=olog,
        config=config,
    )

    decision = await scheduler.run()

    print(f"Decision: {decision.value}")
    print(f"Task retry count: {flaky.retry_count}")
    print(f"Task status: {flaky.status.value}")
    print(f"Total agent calls: {call_count[0]}")
    if flaky.result and flaky.result.success:
        print(f"Output: {flaky.result.output}")

    # Check failure history recorded in metadata
    history = flaky.metadata.get("failure_history", [])
    print(f"Failure history: {len(history)} entries")
    for h in history:
        print(f"  - {h}")


# ---------------------------------------------------------------------------
# Example 6: Dynamic replanning (insert_node_between)
# ---------------------------------------------------------------------------

def example_6_dynamic_replanning():
    """Insert a diagnostic task between two existing tasks."""
    from corecoder.orchestration.models import TaskNode
    from corecoder.orchestration.graph import TaskGraph

    print("\n" + "=" * 60)
    print("EXAMPLE 6: Dynamic Replanning (insert_node_between)")
    print("=" * 60)

    graph = TaskGraph(name="example_6")

    a = TaskNode(id="A", title="Write code", description="Implement feature")
    b = TaskNode(id="B", title="Deploy", description="Ship to production")

    graph.add_node(a)
    graph.add_node(b)
    graph.add_dependency("B", "A")
    a.transition_to("success")  # simulate A completed
    graph.mark_completed("A")

    print(f"Before replan: ready={[n.id for n in graph.get_ready_tasks()]}")

    # Verifier detects: deployment needs a staging step
    diag = TaskNode(
        id="STAGING",
        title="Stage deployment",
        description="Deploy to staging environment and run smoke tests",
        priority=10,
    )
    graph.insert_node_between("A", "B", diag)

    print(f"After insert: A → STAGING → B")
    print(f"Topological order: {[n.id for n in graph.topological_sort()]}")
    print(f"Ready tasks: {[n.id for n in graph.get_ready_tasks()]}")

    # Complete staging
    graph.mark_completed("STAGING")
    print(f"After staging complete — ready: {[n.id for n in graph.get_ready_tasks()]}")


# ---------------------------------------------------------------------------
# Example 7: Verification pipeline
# ---------------------------------------------------------------------------

def example_7_verification():
    """Chain multiple verifiers together."""
    from corecoder.orchestration.models import ExecutionResult, TaskNode
    from corecoder.orchestration.verifier import (
        CompositeVerifier,
        NoOpVerifier,
        OutputVerifier,
        FileExistsVerifier,
    )

    print("\n" + "=" * 60)
    print("EXAMPLE 7: Verification Pipeline")
    print("=" * 60)

    # Build a composite verifier
    verifier = CompositeVerifier()
    verifier.add(NoOpVerifier())
    verifier.add(OutputVerifier())
    verifier.add(FileExistsVerifier())

    # Test 1: Successful output with required patterns
    result1 = ExecutionResult(
        success=True,
        output="Created src/models.py\nSUCCESS: All tests pass\n",
    )
    meta1 = {
        "required_patterns": ["SUCCESS"],
        "forbidden_patterns": ["TODO", "FIXME"],
        "expected_files": ["src/models.py"],
    }
    vr1 = verifier.verify(result1, task_metadata=meta1, working_dir=".")
    print(f"\nTest 1 (should pass): passed={vr1.passed}")
    print(f"  Checks: {vr1.checks_run}")
    print(f"  Failures: {vr1.failures}")

    # Test 2: Missing required pattern
    result2 = ExecutionResult(success=True, output="Created src/models.py\n")
    vr2 = verifier.verify(result2, task_metadata=meta1, working_dir=".")
    print(f"\nTest 2 (should fail - missing SUCCESS): passed={vr2.passed}")
    print(f"  Failures: {vr2.failures}")

    # Test 3: Forbidden pattern found
    result3 = ExecutionResult(success=True, output="SUCCESS\n// TODO: refactor later\n")
    vr3 = verifier.verify(result3, task_metadata=meta1, working_dir=".")
    print(f"\nTest 3 (should fail - TODO found): passed={vr3.passed}")
    print(f"  Failures: {vr3.failures}")


# ---------------------------------------------------------------------------
# Example 8: Persistence (save/load)
# ---------------------------------------------------------------------------

def example_8_persistence():
    """Save a graph to JSON, load it back, verify fidelity."""
    import tempfile
    import os
    from corecoder.orchestration.models import TaskNode
    from corecoder.orchestration.graph import TaskGraph
    from corecoder.orchestration.storage import JSONStorage

    print("\n" + "=" * 60)
    print("EXAMPLE 8: Persistence (Save/Load)")
    print("=" * 60)

    # Build a graph
    graph = TaskGraph(name="persist_test")
    a = TaskNode(id="A", title="Step 1", description="First step")
    b = TaskNode(id="B", title="Step 2", description="Second step")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_dependency("B", "A")
    a.transition_to("success")
    graph.mark_completed("A")

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(base_dir=tmpdir)
        storage.save_graph(graph.to_dict())

        print(f"Saved graph with {graph.node_count} nodes to {tmpdir}")

        # Load
        loaded_data = storage.load_graph()
        loaded_graph = TaskGraph.from_dict(loaded_data)

        print(f"Loaded graph: {loaded_graph}")
        print(f"  Node count: {loaded_graph.node_count}")
        print(f"  A status: {loaded_graph.get_node('A').status.value}")
        print(f"  B status: {loaded_graph.get_node('B').status.value}")
        print(f"  B deps: {loaded_graph.get_dependencies('B')}")
        print(f"  Round-trip fidelity: {graph.node_count == loaded_graph.node_count}")

        # List runs
        runs = storage.list_runs()
        print(f"  Runs: {runs}")


# ---------------------------------------------------------------------------
# Example 9: Working memory injection
# ---------------------------------------------------------------------------

def example_9_working_memory():
    """Build working memory for a task and render it as a prompt."""
    from corecoder.orchestration.models import TaskNode
    from corecoder.orchestration.graph import TaskGraph
    from corecoder.orchestration.memory import MemoryInjector

    print("\n" + "=" * 60)
    print("EXAMPLE 9: Working Memory Injection")
    print("=" * 60)

    graph = TaskGraph(name="memory_test")

    # Setup: task A completed, task B depends on it
    a = TaskNode(
        id="A",
        title="Initialize project",
        description="Create project structure",
        artifacts={"files": ["pyproject.toml", "src/__init__.py"]},
        metadata={"assumptions": ["Python 3.10+ is available"]},
    )
    a.transition_to("success")
    a.result = type('obj', (object,), {"success": True})()

    b = TaskNode(
        id="B",
        title="Write core module",
        description="Implement the main business logic in src/core.py",
    )

    graph.add_node(a)
    graph.add_node(b)
    graph.add_dependency("B", "A")

    # Inject memory for task B
    injector = MemoryInjector()
    memory = injector.build(
        task_id="B",
        graph=graph,
        goal="Build a CLI todo app",
        run_id="demo_run",
        plan_summary="1. Init project → 2. Core module → 3. CLI → 4. Tests",
    )

    prompt = memory.to_prompt_context()
    print(prompt)


# ---------------------------------------------------------------------------
# Example 10: Full Orchestrator pipeline
# ---------------------------------------------------------------------------

async def example_10_full_orchestrator():
    """End-to-end orchestration using the Orchestrator class."""
    from corecoder.orchestration.orchestrator import Orchestrator, OrchestratorConfig
    from corecoder.orchestration.planner import StaticPlanner

    print("\n" + "=" * 60)
    print("EXAMPLE 10: Full Orchestrator Pipeline")
    print("=" * 60)

    # Recipe for a simple coding task
    recipe = [
        {"title": "Create project structure", "description": "Make directories and config files", "priority": 10},
        {"title": "Implement core logic", "description": "Write the main module", "deps": [0], "priority": 8},
        {"title": "Add CLI interface", "description": "Wire up argparse CLI", "deps": [1], "priority": 5},
        {"title": "Write tests", "description": "Add unit tests for core logic", "deps": [1], "priority": 4},
    ]

    orchestrator = Orchestrator(OrchestratorConfig(
        goal="Build a simple CLI calculator",
        continue_on_failure=True,
        auto_persist=True,
        storage_dir=".corecoder/orchestration/examples",
    ))
    orchestrator.set_planner(StaticPlanner(recipe))
    orchestrator.set_agent(_mock_agent_chat)

    result = await orchestrator.run("Build a simple CLI calculator")

    print(f"Success: {result.success}")
    print(f"Tasks: {result.tasks_total} total, "
          f"{result.tasks_succeeded} succeeded, "
          f"{result.tasks_failed} failed")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"Replans: {result.replans_used}")
    print(f"Run ID: {result.run_id}")

    if result.graph:
        for node in result.graph.topological_sort():
            print(f"  [{node.status.value}] {node.title}")


# ---------------------------------------------------------------------------
# Example 11: LLMPlanner — parsing a plan from LLM output
# ---------------------------------------------------------------------------

def example_11_llm_planner_parsing():
    """Demonstrate parsing an LLM plan response into a TaskGraph."""
    from corecoder.orchestration.planner import LLMPlanner

    print("\n" + "=" * 60)
    print("EXAMPLE 11: LLMPlanner Response Parsing")
    print("=" * 60)

    # Simulated LLM response
    llm_response = json.dumps({
        "plan_summary": "Build a REST API with 3 endpoints",
        "assumptions": ["Python 3.10+", "FastAPI framework"],
        "constraints": ["Must use async/await", "No external DB required"],
        "tasks": [
            {
                "title": "Initialize FastAPI project",
                "description": "Create main.py with FastAPI app instance",
                "dependencies": [],
                "priority": 10,
                "verification": {"expected_files": ["main.py"]},
            },
            {
                "title": "Implement GET /items",
                "description": "Create the list endpoint with query params",
                "dependencies": [0],
                "priority": 8,
            },
            {
                "title": "Implement POST /items",
                "description": "Create endpoint with Pydantic validation",
                "dependencies": [0],
                "priority": 8,
            },
            {
                "title": "Add error handling",
                "description": "Global exception handlers",
                "dependencies": [1, 2],
                "priority": 5,
            },
            {
                "title": "Write API tests",
                "description": "Test all endpoints with httpx",
                "dependencies": [3],
                "priority": 4,
            },
        ],
    })

    planner = LLMPlanner()
    result = planner._parse_response(llm_response, "Build a REST API")

    print(f"Plan summary: {result.plan_summary}")
    print(f"Assumptions: {result.assumptions}")
    print(f"Constraints: {result.constraints}")
    print(f"Tasks: {result.graph.node_count}")
    print(f"Topological order:")
    for i, node in enumerate(result.graph.topological_sort()):
        print(f"  {i}. [{node.title}] deps={node.dependencies}")


# ---------------------------------------------------------------------------
# Example 12: Recovery from interrupted execution
# ---------------------------------------------------------------------------

def example_12_recovery_from_interruption():
    """Simulate resuming after a crash — RUNNING tasks reset to PENDING."""
    from corecoder.orchestration.models import TaskNode, TaskStatus
    from corecoder.orchestration.graph import TaskGraph
    from corecoder.orchestration.recovery import resume_graph_state

    print("\n" + "=" * 60)
    print("EXAMPLE 12: Recovery from Interruption")
    print("=" * 60)

    graph = TaskGraph(name="recovery_test")

    a = TaskNode(id="A", title="Completed task", description="Done already")
    a.transition_to("success")
    b = TaskNode(id="B", title="Running when crashed", description="Was mid-execution")
    b.transition_to("running")
    c = TaskNode(id="C", title="Not started", description="Never began")

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)

    print("Before recovery:")
    for node in graph:
        print(f"  {node.id}: {node.status.value}")

    # Simulate recovery
    graph_data = graph.to_dict()
    reset_ids = resume_graph_state(graph, graph_data)

    print(f"\nReset task IDs: {reset_ids}")
    print("After recovery:")
    for node in graph:
        print(f"  {node.id}: {node.status.value}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

async def main():
    """Run all examples."""
    example_1_dag_basics()
    example_2_failure_propagation()
    example_3_static_planner()
    await example_4_full_execution()
    await example_5_retry_and_recovery()
    example_6_dynamic_replanning()
    example_7_verification()
    example_8_persistence()
    example_9_working_memory()
    await example_10_full_orchestrator()
    example_11_llm_planner_parsing()
    example_12_recovery_from_interruption()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
