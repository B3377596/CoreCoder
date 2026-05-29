"""Symbolic retrieval layer ?*the "repository cognition" engine.

Pipeline:
    Task text ?*TaskIntent ?*RetrievalQuery ?*Symbol Routing
    ?*Dependency Expansion ?*Summary Ranking ?*Metadata Fragments

Design: metadata-first, symbolic/structural, no embeddings.
File contents are read by the agent via tools, not by the retriever.
"""

from corecoder.retrieval.models import (
    SymbolInfo,
    FileSummary,
    TaskIntent,
    RetrievalQuery,
    RankedFile,
    RetrievalMeta,
    BidirectionalDepGraph,
)
from corecoder.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.retrieval.summaries import FileSummaryManager
from corecoder.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.retrieval.dependency_graph import build_dependency_graph
from corecoder.retrieval.ranker import StructuredRanker

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
