"""Context Orchestrator — dynamic context assembly for the agent runtime.

Sits between the Scheduler and Executor, assembling context dynamically
based on the current task, execution state, token budget, and repository
structure.  Never sends raw history or full repo to the agent.

Key components:
- ContextOrchestrator: central engine
- ContextFragment: typed, scored context atom
- ContextAssemblyPipeline: collect → rank → deduplicate → compress → budget
- RepositoryContextRetriever: symbolic graph-aware retrieval
- SymbolOwnershipGraph: symbol → file, file → symbols, fuzzy lookup
- FileSummaryManager: heuristic file purpose/responsibility summaries
- TaskIntentAnalyzer: classify task type (bug_fix, cli_change, etc.)
- StructuredRanker: multi-factor scoring with retrieval reasoning
- StatePolicy: per-execution-state context profiles
"""

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextBundle,
    TokenBudget,
    LayerBudget,
    ContextRequest,
    ContextSource,
    ContextType,
    ExecutionState,
)
from corecoder.orchestration.context.layers import (
    ContextLayer,
    SystemContextLayer,
    TaskContextLayer,
    WorkingMemoryContextLayer,
    FailureMemoryContextLayer,
    ConstraintContextLayer,
    ExecutionPolicyContextLayer,
)
from corecoder.orchestration.context.ranker import ContextRanker
from corecoder.orchestration.context.retriever import (
    RepositoryContextRetriever,
    RetrievalOptions,
)
from corecoder.orchestration.context.pipeline import ContextAssemblyPipeline
from corecoder.orchestration.context.policies import (
    StatePolicy,
    get_policy,
    get_retrieval_options,
    POLICIES,
)
from corecoder.orchestration.context.orchestrator import (
    ContextOrchestrator,
    ContextOrchestratorConfig,
    AssemblyResult,
)

# New symbolic retrieval layer
from corecoder.orchestration.retrieval.models import (
    SymbolInfo,
    FileSummary,
    TaskIntent,
    RetrievalQuery,
    RankedFile,
    RetrievalMeta,
)
from corecoder.orchestration.retrieval.symbol_index import SymbolOwnershipGraph
from corecoder.orchestration.retrieval.summaries import FileSummaryManager
from corecoder.orchestration.retrieval.task_intent import TaskIntentAnalyzer
from corecoder.orchestration.retrieval.query_planner import RetrievalQueryPlanner
from corecoder.orchestration.retrieval.ranker import StructuredRanker

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
    "TaskIntentAnalyzer",
    "RetrievalQueryPlanner",
    "StructuredRanker",
    "SymbolInfo",
    "FileSummary",
    "TaskIntent",
    "RetrievalQuery",
    "RankedFile",
    "RetrievalMeta",
]
