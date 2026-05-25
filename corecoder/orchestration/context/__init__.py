"""Context Orchestrator — dynamic context assembly for the agent runtime.

Sits between the Scheduler and Executor, assembling context dynamically
based on the current task, execution state, token budget, and repository
structure.  Never sends raw history or full repo to the agent.

Key components:
- ContextOrchestrator: central engine
- ContextFragment: typed, scored context atom
- ContextAssemblyPipeline: collect → rank → deduplicate → compress → budget
- RepositoryContextRetriever: graph-aware file/symbol retrieval
- ContextRanker: multi-signal relevance scoring
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
    # Ranker
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
]
