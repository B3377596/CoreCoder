"""Data models for Retrieval V2.

The retrieval layer is responsible for turning a user request plus runtime
state into a structured retrieval plan, a repository graph query, and
observable ranking metadata.  These models intentionally separate:

- task understanding: what the user is trying to achieve
- retrieval planning: how we should search the repository
- retrieval context: what execution state should influence search
- repository graph: the normalized codebase abstraction
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentFamily(str, Enum):
    """Legacy high-level routing family kept for backward compatibility."""

    EXECUTION = "execution"
    UNDERSTANDING = "understanding"
    NAVIGATION = "navigation"
    EXPLANATION = "explanation"
    PLANNING = "planning"


class RetrievalMode(str, Enum):
    EXECUTION = "execution"
    UNDERSTANDING = "understanding"
    NAVIGATION = "navigation"
    EXPLANATION = "explanation"
    PLANNING = "planning"


class GraphNodeType(str, Enum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class GraphEdgeType(str, Enum):
    CONTAINS = "contains"
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    REFERENCES = "references"


@dataclass
class TaskEntity:
    name: str
    kind: str = "unknown"
    confidence: float = 0.5
    source: str = ""


@dataclass
class TaskConstraint:
    text: str
    kind: str = "constraint"


@dataclass
class TaskUnderstanding:
    """Semantic understanding of the task without over-committing to labels."""

    goal: str = ""
    objective: str = ""
    entities: list[TaskEntity] = field(default_factory=list)
    constraints: list[TaskConstraint] = field(default_factory=list)
    likely_modules: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class RetrievalPlan:
    """A planned retrieval strategy generated before repository search."""

    task_type: str = "general"
    objective: str = ""
    primary_symbols: list[str] = field(default_factory=list)
    retrieval_scopes: list[str] = field(default_factory=list)
    expansion_depth: int = 1
    retrieval_strategy: str = "balanced"
    target_files: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    plan_reasoning: list[str] = field(default_factory=list)


@dataclass
class RetrievalRequest:
    """An adaptive retrieval follow-up request."""

    reason: str
    additional_scopes: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    requested_files: list[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """Execution-aware retrieval inputs.

    Retrieval is no longer based on user query alone.  It is influenced by the
    active working set, current plan, previous failures, and memory of earlier
    attempts.
    """

    user_query: str = ""
    active_files: list[str] = field(default_factory=list)
    active_symbols: list[str] = field(default_factory=list)
    current_plan: RetrievalPlan | None = None
    working_memory: list[str] = field(default_factory=list)
    previous_failures: list[str] = field(default_factory=list)
    previous_queries: list[str] = field(default_factory=list)
    retrieval_round: int = 1
    requested_more_context: bool = False
    followup_requests: list[RetrievalRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def request_more_context(
        self,
        reason: str,
        additional_scopes: list[str] | None = None,
        missing_symbols: list[str] | None = None,
        requested_files: list[str] | None = None,
    ) -> RetrievalRequest:
        """Signal that retrieval should broaden or refine the search."""
        request = RetrievalRequest(
            reason=reason,
            additional_scopes=list(additional_scopes or []),
            missing_symbols=list(missing_symbols or []),
            requested_files=list(requested_files or []),
        )
        self.requested_more_context = True
        self.retrieval_round += 1
        self.followup_requests.append(request)
        return request


@dataclass
class RepositoryNode:
    id: str
    node_type: GraphNodeType
    name: str
    filepath: str = ""
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryEdge:
    source: str
    target: str
    edge_type: GraphEdgeType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryGraph:
    """Unified repository graph built from symbols.json and dependencies.json."""

    nodes: dict[str, RepositoryNode] = field(default_factory=dict)
    edges: list[RepositoryEdge] = field(default_factory=list)
    adjacency: dict[str, list[RepositoryEdge]] = field(default_factory=lambda: defaultdict(list))
    reverse_adjacency: dict[str, list[RepositoryEdge]] = field(default_factory=lambda: defaultdict(list))
    symbol_to_node_ids: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    file_to_node_ids: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: RepositoryNode) -> None:
        self.nodes[node.id] = node
        if node.node_type != GraphNodeType.FILE:
            self.symbol_to_node_ids[node.name].append(node.id)
        if node.filepath:
            self.file_to_node_ids[node.filepath].append(node.id)
        if node.node_type == GraphNodeType.FILE:
            self.file_to_node_ids[node.name].append(node.id)

    def add_edge(self, edge: RepositoryEdge) -> None:
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge)
        self.reverse_adjacency[edge.target].append(edge)

    def get_node(self, node_id: str) -> RepositoryNode | None:
        return self.nodes.get(node_id)

    def file_node_id(self, filepath: str) -> str | None:
        normalized = filepath.replace("\\", "/")
        return normalized if normalized in self.nodes else None

    def file_nodes(self) -> list[RepositoryNode]:
        return [n for n in self.nodes.values() if n.node_type == GraphNodeType.FILE]

    def symbol_nodes(self, name: str) -> list[RepositoryNode]:
        return [self.nodes[node_id] for node_id in self.symbol_to_node_ids.get(name, []) if node_id in self.nodes]

    def neighbors(
        self,
        node_or_name: str,
        edge_types: set[GraphEdgeType] | None = None,
        reverse: bool = False,
    ) -> list[RepositoryNode]:
        """Return graph neighbors using node id, filepath, or symbol name."""
        start_ids = self._resolve_node_ids(node_or_name)
        edges = self.reverse_adjacency if reverse else self.adjacency
        results: list[RepositoryNode] = []
        seen: set[str] = set()
        for node_id in start_ids:
            for edge in edges.get(node_id, []):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                target_id = edge.source if reverse else edge.target
                if target_id in seen:
                    continue
                target = self.nodes.get(target_id)
                if target is None:
                    continue
                seen.add(target_id)
                results.append(target)
        return results

    def expand(
        self,
        node_or_name: str,
        depth: int = 1,
        edge_types: set[GraphEdgeType] | None = None,
    ) -> set[str]:
        """Expand outward in the graph and return visited node ids."""
        frontier = set(self._resolve_node_ids(node_or_name))
        visited = set(frontier)
        for _ in range(max(0, depth)):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for edge in self.adjacency.get(node_id, []):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    if edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.add(edge.target)
                for edge in self.reverse_adjacency.get(node_id, []):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    if edge.source not in visited:
                        visited.add(edge.source)
                        next_frontier.add(edge.source)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def shortest_path(
        self,
        source: str,
        target: str,
        edge_types: set[GraphEdgeType] | None = None,
    ) -> list[str]:
        """Breadth-first shortest path on the repository graph."""
        from collections import deque

        source_ids = self._resolve_node_ids(source)
        target_ids = set(self._resolve_node_ids(target))
        if not source_ids or not target_ids:
            return []

        queue = deque((sid, [sid]) for sid in source_ids)
        visited = set(source_ids)
        while queue:
            node_id, path = queue.popleft()
            if node_id in target_ids:
                return path
            for edge in self.adjacency.get(node_id, []):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))
            for edge in self.reverse_adjacency.get(node_id, []):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                if edge.source not in visited:
                    visited.add(edge.source)
                    queue.append((edge.source, path + [edge.source]))
        return []

    def related_files(
        self,
        node_or_name: str,
        depth: int = 1,
        edge_types: set[GraphEdgeType] | None = None,
    ) -> list[str]:
        """Expand from a node and return touched repository files."""
        visited = self.expand(node_or_name, depth=depth, edge_types=edge_types)
        files: list[str] = []
        seen: set[str] = set()
        for node_id in visited:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node.node_type == GraphNodeType.FILE:
                filepath = node.name
            else:
                filepath = node.filepath
            if filepath and filepath not in seen:
                seen.add(filepath)
                files.append(filepath)
        return files

    def _resolve_node_ids(self, node_or_name: str) -> list[str]:
        normalized = node_or_name.replace("\\", "/")
        if normalized in self.nodes:
            return [normalized]
        if normalized in self.symbol_to_node_ids:
            return list(self.symbol_to_node_ids[normalized])
        if normalized in self.file_to_node_ids:
            return list(self.file_to_node_ids[normalized])
        # Case-insensitive symbol fallback.
        lowered = normalized.lower()
        matches: list[str] = []
        for name, node_ids in self.symbol_to_node_ids.items():
            if name.lower() == lowered:
                matches.extend(node_ids)
        return matches


@dataclass
class ProjectCognition:
    entrypoints: list[str] = field(default_factory=list)
    major_components: list[str] = field(default_factory=list)
    architecture_summary: str = ""
    execution_flow: list[str] = field(default_factory=list)
    primary_capabilities: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)


@dataclass
class RepositoryTopology:
    architectural_hubs: list[str] = field(default_factory=list)
    leaf_modules: list[str] = field(default_factory=list)
    entrypoint_paths: list[str] = field(default_factory=list)
    centrality_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ArchitecturalCentrality:
    filepath: str
    fan_in: int = 0
    fan_out: int = 0
    is_entrypoint: bool = False
    is_leaf: bool = False
    centrality: float = 0.0


@dataclass
class SymbolInfo:
    name: str
    kind: str
    defined_in: str
    line: int = 0
    signature: str = ""
    doc_brief: str = ""
    exported: bool = False
    parent: str = ""


@dataclass
class FileSummary:
    path: str
    purpose: str = ""
    responsibilities: list[str] = field(default_factory=list)
    key_symbols: list[str] = field(default_factory=list)
    category: str = ""
    file_type: str = ""


@dataclass
class TaskIntent:
    """Legacy compatibility layer around richer task understanding."""

    family: str = ""
    type: str = ""
    symbols: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    entrypoint_related: bool = False
    affected_files: list[str] = field(default_factory=list)
    confidence: float = 0.5
    understanding: TaskUnderstanding | None = None


@dataclass
class RetrievalQuery:
    """Legacy query object preserved for backward compatibility."""

    symbols: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    likely_files: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    expand_dependencies: bool = True
    dependency_radius: int = 1
    plan: RetrievalPlan | None = None


@dataclass
class BidirectionalDepGraph:
    imports: dict[str, list[str]] = field(default_factory=dict)
    imported_by: dict[str, list[str]] = field(default_factory=dict)

    def get_imports(self, filepath: str) -> list[str]:
        return self.imports.get(filepath, [])

    def get_imported_by(self, filepath: str) -> list[str]:
        return self.imported_by.get(filepath, [])

    def neighborhood(self, seed: str, radius: int = 1) -> set[str]:
        result: set[str] = {seed}
        frontier: set[str] = {seed}
        for _ in range(radius):
            next_frontier: set[str] = set()
            for f in frontier:
                for neighbor in self.imports.get(f, []) + self.imported_by.get(f, []):
                    if neighbor not in result:
                        result.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return result


@dataclass
class RankedFile:
    filepath: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    symbol_matches: list[str] = field(default_factory=list)
    summary_match: bool = False
    dependency_neighbor: bool = False
    symbols: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalMeta:
    query: RetrievalQuery = field(default_factory=RetrievalQuery)
    intent: TaskIntent = field(default_factory=TaskIntent)
    understanding: TaskUnderstanding | None = None
    plan: RetrievalPlan | None = None
    retrieval_context: RetrievalContext | None = None
    total_files_considered: int = 0
    total_files_ranked: int = 0
    retrieval_time_ms: float = 0.0
    pipeline_stages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    context_size: int = 0
    token_cost: int = 0

    @staticmethod
    def from_rankings(
        expected: set[str],
        retrieved: list[str],
        token_cost: int = 0,
        ks: tuple[int, ...] = (1, 3, 5),
    ) -> "RetrievalMetrics":
        metrics = RetrievalMetrics(context_size=len(retrieved), token_cost=token_cost)
        for k in ks:
            topk = retrieved[:k]
            metrics.hit_rate_at_k[k] = 1.0 if set(topk) & expected else 0.0
            metrics.recall_at_k[k] = (
                len(set(topk) & expected) / len(expected) if expected else 0.0
            )
        rr = 0.0
        for idx, path in enumerate(retrieved, start=1):
            if path in expected:
                rr = 1.0 / idx
                break
        metrics.mrr = rr
        return metrics
