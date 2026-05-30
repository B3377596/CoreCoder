"""Directed Acyclic Graph for task orchestration.

The TaskGraph is the central data structure of the orchestration layer.
It enforces acyclicity at edge-insertion time (not during traversal) so
that all downstream consumers can assume the graph is valid.

Implementation uses adjacency lists rather than a matrix because task
graphs are sparse  most nodes have 1-3 edges, so O(V+E) algorithms
are practical even for large plans.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from corecoder.agent.dag.models import TaskNode, TaskStatus


class CycleDetectedError(ValueError):
    """Raised when adding a dependency would create a cycle."""

    def __init__(self, from_id: str, to_id: str, path: list[str]):
        self.from_id = from_id
        self.to_id = to_id
        self.path = path
        cycle_str = "  ".join(path)
        super().__init__(
            f"Adding edge {from_id}  {to_id} would create cycle: {cycle_str}"
        )


class TaskGraph:
    """A directed acyclic graph of TaskNodes.

    Nodes are stored by ID.  Edges represent "A depends on B"      task A cannot start until task B completes successfully.

    The graph is NOT thread-safe.  All mutations should happen during
    the planning phase or under the scheduler's lock.
    """

    def __init__(self, name: str = "task_graph"):
        self.name = name
        self._nodes: dict[str, TaskNode] = {}
        # _successors[a] = {b, c} means b and c depend on a
        self._successors: dict[str, set[str]] = {}
        # _predecessors[b] = {a} means b depends on a
        self._predecessors: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # node management
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, TaskNode]:
        return self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def add_node(self, node: TaskNode) -> None:
        """Insert a task node.  Replaces if ID already exists."""
        if node.id not in self._nodes:
            self._successors[node.id] = set()
            self._predecessors[node.id] = set()
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges incident to it."""
        if node_id not in self._nodes:
            return

        # remove outgoing edges
        for succ in list(self._successors.get(node_id, set())):
            self._predecessors[succ].discard(node_id)
        # remove incoming edges
        for pred in list(self._predecessors.get(node_id, set())):
            self._successors[pred].discard(node_id)

        del self._nodes[node_id]
        del self._successors[node_id]
        del self._predecessors[node_id]

    def get_node(self, node_id: str) -> TaskNode | None:
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    # dependency management
    # ------------------------------------------------------------------

    def add_dependency(self, dependent_id: str, prerequisite_id: str) -> None:
        """Declare that `dependent_id` depends on `prerequisite_id`.

        Raises CycleDetectedError if this edge would create a cycle.
        Both nodes must already exist in the graph.
        """
        if dependent_id not in self._nodes:
            raise KeyError(f"Node not found: {dependent_id}")
        if prerequisite_id not in self._nodes:
            raise KeyError(f"Node not found: {prerequisite_id}")
        if dependent_id == prerequisite_id:
            raise CycleDetectedError(
                dependent_id, prerequisite_id, [dependent_id, prerequisite_id]
            )

        # Temporarily add the edge and check for cycles
        self._successors[prerequisite_id].add(dependent_id)
        self._predecessors[dependent_id].add(prerequisite_id)

        cycle = self._find_cycle()
        if cycle is not None:
            # roll back
            self._successors[prerequisite_id].discard(dependent_id)
            self._predecessors[dependent_id].discard(prerequisite_id)
            raise CycleDetectedError(dependent_id, prerequisite_id, cycle)

        # Update the node's own dependency list for serialization
        node = self._nodes[dependent_id]
        if prerequisite_id not in node.dependencies:
            node.dependencies.append(prerequisite_id)
            node.touch()

    def remove_dependency(self, dependent_id: str, prerequisite_id: str) -> None:
        """Remove a dependency edge."""
        self._successors.get(prerequisite_id, set()).discard(dependent_id)
        self._predecessors.get(dependent_id, set()).discard(prerequisite_id)
        node = self._nodes.get(dependent_id)
        if node and prerequisite_id in node.dependencies:
            node.dependencies.remove(prerequisite_id)
            node.touch()

    # ------------------------------------------------------------------
    # graph queries
    # ------------------------------------------------------------------

    def get_dependencies(self, node_id: str) -> list[str]:
        """Return the IDs of all direct prerequisites of `node_id`."""
        return list(self._predecessors.get(node_id, set()))

    def get_dependents(self, node_id: str) -> list[str]:
        """Return the IDs of all nodes that directly depend on `node_id`."""
        return list(self._successors.get(node_id, set()))

    def get_ready_tasks(self) -> list[TaskNode]:
        """Return tasks whose dependencies are all satisfied (SUCCESS).

        A task is ready when:
        - Its status is PENDING
        - All its prerequisite nodes have status SUCCESS
        - It is not BLOCKED or SKIPPED

        Sorted by priority (descending) so the scheduler picks the
        most important ready task first.
        """
        ready: list[TaskNode] = []
        for node in self._nodes.values():
            if not TaskGraph._status_eq(node.status, TaskStatus.PENDING):
                continue
            if self._all_deps_satisfied(node.id):
                ready.append(node)
        ready.sort(key=lambda n: n.priority, reverse=True)
        return ready

    def get_blocked_tasks(self) -> list[TaskNode]:
        """Return PENDING tasks that cannot run because deps are unsatisfied."""
        blocked: list[TaskNode] = []
        for node in self._nodes.values():
            if not TaskGraph._status_eq(node.status, TaskStatus.PENDING):
                continue
            if not self._all_deps_satisfied(node.id):
                blocked.append(node)
        return blocked

    def get_failed_tasks(self) -> list[TaskNode]:
        """Return all tasks with FAILED status."""
        return [n for n in self._nodes.values() if TaskGraph._status_eq(n.status, TaskStatus.FAILED)]

    def get_terminal_tasks(self) -> list[TaskNode]:
        """Return tasks with no dependents (leaf nodes in the DAG)."""
        return [
            n
            for n in self._nodes.values()
            if not self._successors.get(n.id, set())
        ]

    def get_root_tasks(self) -> list[TaskNode]:
        """Return tasks with no dependencies (source nodes in the DAG)."""
        return [
            n
            for n in self._nodes.values()
            if not self._predecessors.get(n.id, set())
        ]

    def is_complete(self) -> bool:
        """True when every node has reached a terminal state."""
        return all(
            node.status.is_terminal for node in self._nodes.values()
        )

    def is_successful(self) -> bool:
        """True when every node is SUCCESS."""
        return all(
            TaskGraph._status_eq(node.status, TaskStatus.SUCCESS) for node in self._nodes.values()
        )

    def progress(self) -> tuple[int, int, int, int]:
        """Return (success, failed, running, pending) counts."""
        success = sum(1 for n in self._nodes.values() if n.status == TaskStatus.SUCCESS)
        failed = sum(1 for n in self._nodes.values() if TaskGraph._status_eq(n.status, TaskStatus.FAILED))
        running = sum(1 for n in self._nodes.values() if TaskGraph._status_eq(n.status, TaskStatus.RUNNING))
        pending = sum(
            1
            for n in self._nodes.values()
            if n.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.BLOCKED)
        )
        return success, failed, running, pending

    def topological_sort(self) -> list[TaskNode]:
        """Return nodes in topological order (dependencies before dependents).

        Uses Kahn's algorithm.  Returns all nodes if the graph is acyclic,
        otherwise returns a partial ordering up to the first cycle.

        This is primarily useful for serial execution plans and for
        human-readable display of the plan.
        """
        in_degree: dict[str, int] = {
            nid: len(preds) for nid, preds in self._predecessors.items()
        }
        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        result: list[TaskNode] = []

        while queue:
            nid = queue.popleft()
            result.append(self._nodes[nid])
            for succ in self._successors.get(nid, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return result

    def detect_cycles(self) -> list[list[str]] | None:
        """Return all cycles found, or None if the graph is acyclic.

        Uses iterative DFS with coloring (white/gray/black).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}
        cycles: list[list[str]] = []

        def dfs(node_id: str, path: list[str]) -> None:
            color[node_id] = GRAY
            path.append(node_id)
            for succ in self._successors.get(node_id, set()):
                if color[succ] == GRAY:
                    # found a cycle  extract the portion from succ to end
                    cycle_start = path.index(succ)
                    cycles.append(path[cycle_start:] + [succ])
                elif color[succ] == WHITE:
                    dfs(succ, path)
            path.pop()
            color[node_id] = BLACK

        for nid in self._nodes:
            if color[nid] == WHITE:
                dfs(nid, [])

        return cycles if cycles else None

    # ------------------------------------------------------------------
    # graph mutation helpers
    # ------------------------------------------------------------------

    def mark_completed(self, node_id: str, result=None) -> None:
        """Mark a task as SUCCESS.  Updates node result if provided."""
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node not found: {node_id}")
        if result is not None:
            node.result = result
        node.transition_to(TaskStatus.SUCCESS)

    def mark_failed(self, node_id: str, error: str = "") -> None:
        """Mark a task as FAILED and propagate BLOCKED to dependents.

        When a node fails, all downstream nodes that depend on it
        (directly or transitively) are marked BLOCKED because they
        can never become READY until this dependency is resolved.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node not found: {node_id}")
        node.error = error
        node.transition_to(TaskStatus.FAILED)

        # Propagate BLOCKED downstream
        self._propagate_blocked(node_id, set())

    def unblock_dependents(self, node_id: str) -> None:
        """After a previously-failed node becomes SUCCESS, unblock dependents.

        Only unblocks nodes whose ALL dependencies are now satisfied.
        """
        for dep_id in self._successors.get(node_id, set()):
            dep = self._nodes.get(dep_id)
            if dep is None:
                continue
            if TaskGraph._status_eq(dep.status, TaskStatus.BLOCKED) and self._all_deps_satisfied(dep_id):
                dep.transition_to(TaskStatus.PENDING)

    def insert_node_between(
        self, predecessor_id: str, successor_id: str, new_node: TaskNode
    ) -> None:
        """Insert a new node into an existing dependency chain.

        Before:  predecessor  successor
        After:   predecessor  new_node  successor

        This is the primary mechanism for dynamic replanning  when the
        verifier detects a missing step, the planner can inject a new
        node without rebuilding the entire graph.
        """
        self.add_node(new_node)
        # Remove old edge
        self.remove_dependency(successor_id, predecessor_id)
        # Add new edges
        self.add_dependency(new_node.id, predecessor_id)
        self.add_dependency(successor_id, new_node.id)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full graph for persistence."""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [
                {"from": pred, "to": succ}
                for pred, succs in self._successors.items()
                for succ in succs
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskGraph:
        """Restore a graph from serialized state."""
        from corecoder.agent.dag.models import TaskNode as TN

        graph = cls(name=data.get("name", "task_graph"))
        for node_data in data.get("nodes", []):
            node = TN.from_dict(node_data)
            graph.add_node(node)
        for edge in data.get("edges", []):
            # Use internal edge insertion that skips cycle checking
            # (the graph was valid when serialized)
            pred, succ = edge["from"], edge["to"]
            graph._successors[pred].add(succ)
            graph._predecessors[succ].add(pred)
            node = graph._nodes[succ]
            if pred not in node.dependencies:
                node.dependencies.append(pred)
        return graph

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _status_eq(node_status, target: TaskStatus) -> bool:
        """Compare a node's status (which may be a string from deserialization)
        against a TaskStatus enum value."""
        if isinstance(node_status, str):
            return node_status == target.value
        return node_status == target

    def _all_deps_satisfied(self, node_id: str) -> bool:
        """Check if every prerequisite of `node_id` is SUCCESS."""
        for pred_id in self._predecessors.get(node_id, set()):
            pred = self._nodes.get(pred_id)
            if pred is None or not self._status_eq(pred.status, TaskStatus.SUCCESS):
                return False
        return True

    def _propagate_blocked(self, failed_id: str, visited: set[str]) -> None:
        """Recursively mark downstream nodes as BLOCKED."""
        if failed_id in visited:
            return
        visited.add(failed_id)
        for dep_id in self._successors.get(failed_id, set()):
            dep = self._nodes.get(dep_id)
            if dep is None:
                continue
            if dep.status not in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED):
                dep.transition_to(TaskStatus.BLOCKED)
                self._propagate_blocked(dep_id, visited)

    def _find_cycle(self) -> list[str] | None:
        """Find one cycle via DFS.  Returns the cycle path or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}

        def dfs(node_id: str, path: list[str]) -> list[str] | None:
            color[node_id] = GRAY
            path.append(node_id)
            for succ in self._successors.get(node_id, set()):
                if color[succ] == GRAY:
                    cycle_start = path.index(succ)
                    return path[cycle_start:] + [succ]
                elif color[succ] == WHITE:
                    result = dfs(succ, path)
                    if result is not None:
                        return result
            path.pop()
            color[node_id] = BLACK
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                cycle = dfs(nid, [])
                if cycle is not None:
                    return cycle
        return None

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[TaskNode]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        s, f, r, p = self.progress()
        return (
            f"TaskGraph(name={self.name!r}, nodes={len(self._nodes)}, "
            f"success={s}, failed={f}, running={r}, pending={p})"
        )
