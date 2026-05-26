"""Symbolic retrieval layer — the "repository cognition" engine.

Pipeline:
    Task text → TaskIntent → RetrievalQuery → Symbol Routing
    → Dependency Expansion → Summary Ranking → Metadata Fragments

Design: metadata-first, symbolic/structural, no embeddings.
File contents are read by the agent via tools, not by the retriever.
"""

from corecoder.orchestration.retrieval.models import (
    SymbolInfo,
    FileSummary,
    TaskIntent,
    RetrievalQuery,
    RankedFile,
    RetrievalMeta,
    BidirectionalDepGraph,
)
from corecoder.orchestration.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.orchestration.retrieval.summaries import FileSummaryManager
from corecoder.orchestration.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.orchestration.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.orchestration.retrieval.dependency_graph import build_dependency_graph
from corecoder.orchestration.retrieval.ranker import StructuredRanker

__all__ = [
    "SymbolInfo",
    "FileSummary",
    "TaskIntent",
    "RetrievalQuery",
    "RankedFile",
    "RetrievalMeta",
    "BidirectionalDepGraph",
    "SymbolOwnershipGraph",
    "FileSummaryManager",
    "TaskIntentAnalyzer",
    "RetrievalQueryPlanner",
    "build_dependency_graph",
    "StructuredRanker",
]
