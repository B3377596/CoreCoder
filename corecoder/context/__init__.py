"""Context Orchestrator  dynamic context assembly for the staged runtime.

Assembles context dynamically based on the current task, execution state,
token budget, and repository structure. Never sends raw history or full repo
to the agent.

Key components:
- ContextOrchestrator: central engine
- ContextFragment: typed, scored context atom
- ContextAssemblyPipeline: collect  rank  deduplicate  compress  budget
- RepositoryContextRetriever: symbolic graph-aware retrieval
- SymbolOwnershipGraph: symbol  file, file  symbols, fuzzy lookup
- FileSummaryManager: heuristic file purpose/responsibility summaries
- TaskUnderstandingAnalyzer: understand goal, entities, and likely modules
- StructuredRanker: multi-factor scoring with retrieval reasoning
- StatePolicy: per-execution-state context profiles
"""

from corecoder.context.models import (
    ContextFragment,
    ContextBundle,
    TokenBudget,
    LayerBudget,
    ContextRequest,
    ContextSource,
    ContextType,
    ExecutionState,
)
from corecoder.context.layers import (
    ContextLayer,
    SystemContextLayer,
    TaskContextLayer,
    WorkingMemoryContextLayer,
    FailureMemoryContextLayer,
    ConstraintContextLayer,
    ExecutionPolicyContextLayer,
)
from corecoder.context.ranker import ContextRanker
from corecoder.context.retriever import (
    RepositoryContextRetriever,
    RetrievalOptions,
)
from corecoder.context.pipeline import ContextAssemblyPipeline
from corecoder.context.policies import (
    StatePolicy,
    get_policy,
    get_retrieval_options,
    POLICIES,
)
from corecoder.context.orchestrator import (
    ContextOrchestrator,
    ContextOrchestratorConfig,
    AssemblyResult,
)

# New symbolic retrieval layer
from corecoder.retrieval.models import (
    SymbolInfo,
    FileSummary,
    TaskUnderstanding,
    RetrievalQuery,
    RankedFile,
    RetrievalMeta,
)
from corecoder.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.retrieval.summaries import FileSummaryManager
from corecoder.retrieval.task_understanding import TaskUnderstandingAnalyzer
from corecoder.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.retrieval.ranker import StructuredRanker

__all__ = [
    # Models
    "ContextFragment",
    "ContextBundle",
    "TokenBudget",
    "LayerBudget",
    "ContextRequest",
    "ContextSource",
    "ContextType",
    "ExecutionState",
    # Layers
    "ContextLayer",
    "SystemContextLayer",
    "TaskContextLayer",
    "WorkingMemoryContextLayer",
    "FailureMemoryContextLayer",
    "ConstraintContextLayer",
    # Ranker (pipeline)
    "ContextRanker",
    # Retriever
    "RepositoryContextRetriever",
    "RetrievalOptions",
    # Pipeline
    "ContextAssemblyPipeline",
    # Policies
    "StatePolicy",
    "get_policy",
    "get_retrieval_options",
    "POLICIES",
    # Orchestrator
    "ContextOrchestrator",
    "ContextOrchestratorConfig",
    "AssemblyResult",
    # Symbolic retrieval layer
    "SymbolOwnershipGraph",
    "FileSummaryManager",
    "TaskUnderstandingAnalyzer",
    "RetrievalQueryPlanner",
    "StructuredRanker",
    "SymbolInfo",
    "FileSummary",
    "TaskUnderstanding",
    "RetrievalQuery",
    "RankedFile",
    "RetrievalMeta",
]
