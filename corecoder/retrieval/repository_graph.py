"""Repository Graph abstraction for Retrieval V2."""

from __future__ import annotations

from typing import Any

from corecoder.retrieval.models import (
    GraphEdgeType,
    GraphNodeType,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)


def build_repository_graph(
    symbols_json: dict[str, Any],
    dependencies_json: dict[str, Any],
) -> RepositoryGraph:
    """Build a RepositoryGraph from symbols.json and dependencies.json.

    The builder is intentionally tolerant of both legacy and V2 index formats.
    """
    graph = RepositoryGraph()

    for filepath, symbols in symbols_json.items():
        normalized = filepath.replace("\\", "/")
        graph.add_node(
            RepositoryNode(
                id=normalized,
                node_type=GraphNodeType.FILE,
                name=normalized,
                filepath=normalized,
            )
        )

        if not isinstance(symbols, dict):
            continue

        for symbol_name, info in symbols.items():
            if isinstance(info, dict):
                kind = str(info.get("kind", "unknown"))
                line = int(info.get("line", 0) or 0)
                signature = info.get("signature", "")
                parent = info.get("parent", "")
                methods = info.get("methods", [])
                bases = info.get("bases", [])
            else:
                kind = "class" if info else "function"
                line = 0
                signature = ""
                parent = ""
                methods = info if isinstance(info, list) else []
                bases = []

            node_type = _symbol_kind_to_node_type(kind)
            symbol_id = f"{normalized}::{symbol_name}"
            graph.add_node(
                RepositoryNode(
                    id=symbol_id,
                    node_type=node_type,
                    name=symbol_name,
                    filepath=normalized,
                    line=line,
                    metadata={
                        "signature": signature,
                        "parent": parent,
                        "kind": kind,
                    },
                )
            )
            graph.add_edge(
                RepositoryEdge(
                    source=normalized,
                    target=symbol_id,
                    edge_type=GraphEdgeType.CONTAINS,
                )
            )

            for method in methods or []:
                method_name = method if isinstance(method, str) else method.get("name", "")
                if not method_name:
                    continue
                method_id = f"{symbol_id}.{method_name}"
                graph.add_node(
                    RepositoryNode(
                        id=method_id,
                        node_type=GraphNodeType.METHOD,
                        name=method_name,
                        filepath=normalized,
                        metadata={"parent": symbol_name},
                    )
                )
                graph.add_edge(
                    RepositoryEdge(
                        source=symbol_id,
                        target=method_id,
                        edge_type=GraphEdgeType.CONTAINS,
                    )
                )

            for base in bases or []:
                for base_node in graph.symbol_nodes(base):
                    graph.add_edge(
                        RepositoryEdge(
                            source=symbol_id,
                            target=base_node.id,
                            edge_type=GraphEdgeType.INHERITS,
                        )
                    )

    import_edges = dependencies_json.get("resolved_internal_imports") or dependencies_json.get("internal_imports", {})
    for source_file, imports in import_edges.items():
        source_id = source_file.replace("\\", "/")
        if source_id not in graph.nodes:
            continue
        for imported in imports:
            target_file = imported.replace("\\", "/")
            if target_file not in graph.nodes:
                continue
            graph.add_edge(
                RepositoryEdge(
                    source=source_id,
                    target=target_file,
                    edge_type=GraphEdgeType.IMPORTS,
                )
            )

    references = dependencies_json.get("symbol_references", {})
    for source_file, names in references.items():
        source_id = source_file.replace("\\", "/")
        if source_id not in graph.nodes:
            continue
        for name in names:
            for target_node in graph.symbol_nodes(name):
                graph.add_edge(
                    RepositoryEdge(
                        source=source_id,
                        target=target_node.id,
                        edge_type=GraphEdgeType.REFERENCES,
                    )
                )

    call_map = dependencies_json.get("symbol_calls", {})
    for source_file, calls in call_map.items():
        source_id = source_file.replace("\\", "/")
        if source_id not in graph.nodes:
            continue
        for name in calls:
            for target_node in graph.symbol_nodes(name):
                graph.add_edge(
                    RepositoryEdge(
                        source=source_id,
                        target=target_node.id,
                        edge_type=GraphEdgeType.CALLS,
                    )
                )

    return graph


def _symbol_kind_to_node_type(kind: str) -> GraphNodeType:
    lowered = kind.lower()
    if lowered == "class":
        return GraphNodeType.CLASS
    if lowered == "method":
        return GraphNodeType.METHOD
    return GraphNodeType.FUNCTION
