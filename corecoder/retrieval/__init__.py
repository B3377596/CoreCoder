"""Symbolic retrieval layer ?*the "repository cognition" engine.

Pipeline:
    Task text ?*TaskIntent ?*RetrievalQuery ?*Symbol Routing
    ?*Dependency Expansion ?*Summary Ranking ?*Metadata Fragments

Design: metadata-first, symbolic/structural, no embeddings.
File contents are read by the agent via tools, not by the retriever.
"""

from corecoder.retrieval.models import (
    GraphEdgeType,
    GraphNodeType,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
    SymbolInfo,
    FileSummary,
    TaskConstraint,
    TaskEntity,
    TaskUnderstanding,
    TaskIntent,
    RetrievalPlan,
    RetrievalContext,
    RetrievalRequest,
    RetrievalQuery,
    RankedFile,
    RetrievalMeta,
    RetrievalMetrics,
    BidirectionalDepGraph,
)
from corecoder.retrieval.repository_graph import build_repository_graph
from corecoder.retrieval.evaluation import summarize_metrics
from corecoder.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.retrieval.summaries import FileSummaryManager
from corecoder.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.retrieval.retrieval_planner import RetrievalPlanner
from corecoder.retrieval.dependency_graph import build_dependency_graph
from corecoder.retrieval.ranker import StructuredRanker

__all__ = [
    "GraphEdgeType",
    "GraphNodeType",
    "RepositoryEdge",
    "RepositoryGraph",
    "RepositoryNode",
    "SymbolInfo",
    "FileSummary",
    "TaskConstraint",
    "TaskEntity",
    "TaskUnderstanding",
    "TaskIntent",
    "RetrievalPlan",
    "RetrievalContext",
    "RetrievalRequest",
    "RetrievalQuery",
    "RankedFile",
    "RetrievalMeta",
    "RetrievalMetrics",
    "BidirectionalDepGraph",
    "build_repository_graph",
    "summarize_metrics",
    "SymbolOwnershipGraph",
    "FileSummaryManager",
    "TaskIntentAnalyzer",
    "RetrievalPlanner",
    "RetrievalQueryPlanner",
    "build_dependency_graph",
    "StructuredRanker",
]
