# Retrieval V2 Architecture

## Overview

Retrieval V2 upgrades CoreCoder's repository retrieval layer from a one-shot,
query-only heuristic pipeline into a state-aware planning system.

The old flow was:

```text
user query
-> intent classification
-> symbol extraction
-> dependency expansion
-> ranking
-> context build
```

The new flow is:

```text
user query + runtime state
-> task understanding
-> retrieval planning
-> repository graph retrieval
-> structured ranking
-> adaptive evaluation
-> context fragments
```

This refactor introduces five new architectural ideas:

1. `TaskUnderstanding` replaces brittle task over-classification.
2. `RetrievalPlan` becomes the explicit planning artifact before retrieval.
3. `RepositoryGraph` becomes the single graph abstraction over repository structure.
4. `RetrievalContext` makes retrieval state-aware instead of query-only.
5. Adaptive retrieval allows retrieve -> execute -> evaluate -> retrieve again.

## Architecture Diagram

```text
                           +----------------------+
                           |   ContextRequest     |
                           |----------------------|
                           | query / goal         |
                           | active_files         |
                           | focus_symbols        |
                           | working_memory       |
                           | recent_errors        |
                           +----------+-----------+
                                      |
                                      v
                     +--------------------------------------+
                     | TaskIntentAnalyzer.understand()      |
                     |--------------------------------------|
                     | outputs TaskUnderstanding            |
                     | - objective                          |
                     | - entities                           |
                     | - constraints                        |
                     | - likely_modules                     |
                     +----------------+---------------------+
                                      |
                                      v
                     +--------------------------------------+
                     | RetrievalPlanner.plan()              |
                     |--------------------------------------|
                     | outputs RetrievalPlan                |
                     | - primary_symbols                    |
                     | - retrieval_scopes                   |
                     | - expansion_depth                    |
                     | - retrieval_strategy                 |
                     +----------------+---------------------+
                                      |
                                      v
                     +--------------------------------------+
                     | RepositoryContextRetriever           |
                     |--------------------------------------|
                     | builds RetrievalContext              |
                     | collects candidates via graph        |
                     | ranks files                          |
                     | adapts if context is insufficient    |
                     +----------------+---------------------+
                                      |
                                      v
                     +--------------------------------------+
                     | RepositoryGraph                      |
                     |--------------------------------------|
                     | file / class / function / method     |
                     | imports / calls / inherits / refs    |
                     +----------------+---------------------+
                                      |
                                      v
                     +--------------------------------------+
                     | ContextFragment + RetrievalMeta      |
                     +--------------------------------------+
```

## Data Flow

### 1. Index Build

`corecoder/codebase/indexing/index.py`

The repository index builder now emits richer metadata:

- `symbols.json`
  - file -> symbol -> `{kind, line, signature, doc, methods, bases}`
- `dependencies.json`
  - `declared`
  - `imports`
  - `internal_imports`
  - `resolved_internal_imports`
  - `symbol_calls`
  - `symbol_references`
  - `inheritance`

This is the source of truth for Retrieval V2 graph construction.

### 2. Graph Construction

`corecoder/retrieval/repository_graph.py`

`build_repository_graph(symbols_json, dependencies_json)` normalizes index data
into a `RepositoryGraph`.

Nodes:

- `file`
- `class`
- `function`
- `method`

Edges:

- `contains`
- `imports`
- `calls`
- `inherits`
- `references`

The retrieval layer should no longer reason over raw JSON structure directly
when performing structural expansion.

### 3. Task Understanding

`corecoder/retrieval/task_intent.py`

Retrieval V2 weakens the old "hard classify everything into bug_fix /
feature_addition / repo_understanding" style.

Instead, `TaskIntentAnalyzer.understand()` extracts:

- `goal`
- `objective`
- `entities`
- `constraints`
- `likely_modules`
- `query_terms`

This semantic representation preserves more task information and avoids early
loss of detail.

`analyze()` still exists as a compatibility shim for older ranker and retriever
paths.

### 4. Retrieval Planning

`corecoder/retrieval/retrieval_planner.py`

`RetrievalPlanner` converts `TaskUnderstanding + RetrievalContext` into a
`RetrievalPlan`.

Example:

```python
RetrievalPlan(
    task_type="targeted_change",
    objective="modify login logic",
    primary_symbols=["AuthController", "AuthService"],
    retrieval_scopes=["auth", "session"],
    expansion_depth=2,
    retrieval_strategy="task_execution",
)
```

This plan is intentionally retrieval-only. It does not execute anything and it
does not own ranking logic.

### 5. Candidate Collection and Ranking

`corecoder/context/retriever.py`

The retriever now:

1. Builds `TaskUnderstanding`
2. Builds `RetrievalContext`
3. Generates `RetrievalPlan`
4. Converts plan to legacy `RetrievalQuery` for compatibility
5. Collects candidates from:
   - active files
   - symbol ownership graph
   - repository graph expansion
   - likely file hints
   - summary/module scope matching
   - adaptive follow-up requests
6. Applies structured ranking
7. Expands by dependency neighborhood if needed
8. Emits `ContextFragment` and `RetrievalMeta`

### 6. Adaptive Retrieval

Adaptive retrieval is now part of the runtime contract.

If initial retrieval is low confidence or misses symbols, retrieval can trigger:

```python
retrieval_context.request_more_context(
    reason="low_confidence_initial_retrieval",
    additional_scopes=[...],
    missing_symbols=[...],
    requested_files=[...],
)
```

Then the system replans and retrieves again.

This changes retrieval from a one-shot stage into an iterative control loop.

## Class Diagram

```text
TaskUnderstanding
  - goal
  - objective
  - entities
  - constraints
  - likely_modules
  - query_terms

RetrievalContext
  - user_query
  - active_files
  - active_symbols
  - current_plan
  - working_memory
  - previous_failures
  - previous_queries
  - followup_requests
  + request_more_context()

RetrievalPlan
  - task_type
  - objective
  - primary_symbols
  - retrieval_scopes
  - expansion_depth
  - retrieval_strategy
  - target_files
  - required_context

RepositoryGraph
  - nodes
  - edges
  + neighbors()
  + expand()
  + shortest_path()
  + related_files()

RetrievalMetrics
  - hit_rate_at_k
  - recall_at_k
  - mrr
  - context_size
  - token_cost
```

## Module Responsibilities

### `corecoder/codebase/indexing/index.py`

- scan repository files
- parse Python symbols and relations
- resolve internal import structure
- write `symbols.json` and `dependencies.json`

### `corecoder/retrieval/models.py`

- define V2 retrieval primitives
- define graph model
- define state-aware retrieval context
- define benchmark metrics
- preserve legacy compatibility models during migration

### `corecoder/retrieval/task_intent.py`

- semantic task understanding
- lightweight compatibility family inference
- entity / constraint / module extraction

### `corecoder/retrieval/retrieval_planner.py`

- convert understanding + runtime state into a retrieval plan
- choose retrieval strategy and expansion depth
- incorporate adaptive follow-up requests

### `corecoder/retrieval/repository_graph.py`

- normalize repository index data into graph form
- expose graph-native traversal entry point for retrieval

### `corecoder/retrieval/query_planner.py`

- compatibility adapter
- converts `RetrievalPlan` into legacy `RetrievalQuery`
- allows gradual migration of ranker and retriever internals

### `corecoder/context/retriever.py`

- orchestrate the retrieval pipeline
- build retrieval context from task state
- collect candidates
- run ranking
- trigger adaptive retrieval
- emit retrieval metadata

### `corecoder/retrieval/evaluation.py`

- aggregate `RetrievalMetrics`
- support benchmark reporting and comparison

## Differences From V1

### V1

- retrieval started directly from user query
- intent classification was the primary routing artifact
- retrieval was one-shot
- retrieval did not consume task runtime state
- JSON index files were effectively the retrieval API

### V2

- retrieval starts from `TaskUnderstanding`
- `RetrievalPlan` is explicit and inspectable
- retrieval can adapt after failure or low confidence
- retrieval is influenced by active files, memory, failures, and current plan
- `RepositoryGraph` is the structural abstraction layer
- metrics are first-class and benchmark-ready

## Why This Refactor Matters

This architecture makes retrieval much closer to how a real code agent should
operate:

- planning before searching
- using execution state to guide search
- broadening or refocusing when evidence is insufficient
- grounding repository knowledge in a graph instead of ad hoc JSON access

That gives us better extensibility than adding more keyword rules into the old
pipeline.

## Extension Directions

### 1. Replace Compatibility `TaskIntent`

The current stack still carries `TaskIntent` and `RetrievalQuery` as migration
artifacts. A future cleanup can let the ranker and retriever consume
`TaskUnderstanding` and `RetrievalPlan` directly.

### 2. Planner-Aware Retrieval Policies

Different execution planner nodes may require different retrieval plans:

- code modification
- root cause investigation
- verification-only steps
- architecture comprehension

This can be expressed with planner-provided retrieval hints.

### 3. Graph-Weighted Ranking

Current ranking still uses the existing structured ranker. A next step is to
add graph-native signals:

- path distance to active files
- import fan-in / fan-out
- symbol centrality
- shortest-path relevance

### 4. Multi-Round Retrieval Budgeting

Adaptive retrieval currently triggers on low confidence. Future work can make
this budget-aware:

- max retrieval rounds
- token budget per round
- context compression after each round

### 5. Richer Repository Graph

The graph can be extended with:

- test-to-source links
- config-to-runtime links
- package / module namespace nodes
- ownership or churn metadata

### 6. Retrieval Evaluation Benchmarks

`RetrievalMetrics` is intentionally simple and local-first. It can later power:

- per-intent benchmark slices
- per-module recall reports
- regression dashboards
- online retrieval quality telemetry

## Migration Notes

This refactor is intentionally incremental.

Compatibility layers currently retained:

- `TaskIntent`
- `RetrievalQuery`
- `RetrievalQueryPlanner`
- existing `StructuredRanker`

This allows Retrieval V2 architecture to land without forcing a full rewrite of
the ranking and context assembly layers in the same change set.
