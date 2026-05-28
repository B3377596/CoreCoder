"""Task intent analysis — understand WHAT the user is trying to do.

Two-layer classification:
  1. IntentFamily  — high-level: EXECUTION, UNDERSTANDING, NAVIGATION, ...
  2. TaskType      — fine-grained (only for EXECUTION family)

The family determines the retrieval MODE.  Understanding queries go
through an architecture/overview pipeline; execution queries go through
the symbol/task pipeline.  They are fundamentally different systems.

Analysis is purely heuristic — no LLM calls.
"""

from __future__ import annotations

import re
from corecoder.orchestration.retrieval.models import TaskIntent, IntentFamily


class TaskIntentAnalyzer:
    """Two-layer intent classifier: family first, then task type.

    Usage:
        analyzer = TaskIntentAnalyzer()
        intent = analyzer.analyze("这个项目在干什么")
        # → TaskIntent(family=UNDERSTANDING, type="overview", concepts=["architecture", ...])
    """

    # ---- Family classification patterns ----
    # Order matters: more specific patterns checked first within each family.

    _UNDERSTANDING_PATTERNS: list[str] = [
        # Chinese understanding queries
        "干什么", "是什么", "做什么", "有什么用", "怎么工作",
        "介绍一下", "介绍", "概述", "总结", "说明",
        "架构", "结构", "整体", "总体",
        # English understanding queries
        "what does this", "what is this", "how does this work",
        "explain the project", "explain the codebase",
        "explain the architecture", "explain this project",
        "project overview", "architecture overview",
        "understand", "overview", "summarize", "describe",
        "architecture", "what are the", "what is the purpose",
        "how is this organized", "how is the code", "structure",
        "onboarding", "getting started", "walk me through",
        "tell me about", "give me an overview",
    ]

    _NAVIGATION_PATTERNS: list[str] = [
        "where is", "find the", "locate", "show me",
        "which file", "file path", "where does",
        "在哪里", "找", "哪个文件", "定位",
    ]

    _EXPLANATION_PATTERNS: list[str] = [
        "how does", "why does", "how do", "why is",
        "explain how", "explain why", "deep dive",
        "怎么做到", "为什么", "如何实现",
    ]

    _PLANNING_PATTERNS: list[str] = [
        "what do I need", "how should I", "what's the best way",
        "plan", "approach", "strategy", "steps to",
        "需要什么", "怎么做", "如何做", "第一步",
    ]

    # Execution patterns are the DEFAULT — if nothing else matches,
    # the query is assumed to be a task execution request.

    # ---- Task-type patterns (only used for EXECUTION family) ----
    _TYPE_PATTERNS: list[tuple[str, list[str]]] = [
        ("bug_fix", ["fix", "bug", "error", "crash", "broken", "issue", "defect",
                     "fail", "incorrect", "wrong", "not working", "doesn't work"]),
        ("dependency_change", ["install", "uninstall", "upgrade", "downgrade",
                                "dependency", "package", "pip", "uv add", "uv remove",
                                "requirements", "import ", "library"]),
        ("cli_change", ["cli", "command line", "argparse", "click", "typer",
                        "entry point", "entrypoint", "argument", "flag", "option",
                        "terminal", "console", "shell"]),
        ("refactor", ["refactor", "clean up", "cleanup", "restructure",
                      "reorganize", "extract", "split", "merge", "simplify"]),
        ("rename", ["rename", "move to", "relocate", "change name"]),
        ("test_addition", ["test", "unittest", "pytest", "coverage", "assert",
                           "mock", "fixture"]),
        ("feature_integration", ["integrate", "connect", "wire", "hook up",
                                 "combine", "plug into", "interface with"]),
        ("feature_addition", ["add", "new", "create", "implement", "build",
                              "introduce", "support for"]),
        ("documentation", ["document", "docstring", "comment", "readme",
                           "explain", "describe"]),
    ]

    # Understanding-specific subtypes
    _UNDERSTANDING_SUBTYPES: dict[str, list[str]] = {
        "overview": ["干什么", "是什么", "做什么", "overview", "what does this",
                     "what is this", "summarize", "describe the project"],
        "architecture": ["架构", "结构", "architecture", "how is this organized",
                         "how is the code", "design", "pattern"],
        "capabilities": ["有什么用", "capabilities", "features", "what can",
                         "functionality", "purpose"],
        "components": ["modules", "components", "packages", "directories",
                       "模块", "组件"],
    }

    # ---- Semantic query → concept mapping ----
    # Maps natural-language understanding queries to architectural concepts.
    # Handles Chinese, paraphrases, and open-ended questions.
    _SEMANTIC_CONCEPT_MAP: dict[str, list[str]] = {
        # Chinese understanding
        "干什么": ["architecture", "overview", "capabilities", "entrypoint", "purpose"],
        "是什么": ["architecture", "overview", "capabilities", "purpose"],
        "做什么": ["capabilities", "overview", "entrypoint"],
        "有什么用": ["capabilities", "purpose", "overview"],
        "怎么工作": ["architecture", "execution_flow", "entrypoint"],
        "架构": ["architecture", "components", "entrypoint", "structure"],
        "结构": ["architecture", "components", "structure"],
        "模块": ["components", "modules", "architecture"],
        "介绍一下": ["overview", "capabilities", "architecture", "entrypoint"],
        "介绍": ["overview", "capabilities"],

        # English understanding
        "overview": ["architecture", "overview", "entrypoint", "capabilities"],
        "architecture": ["architecture", "components", "entrypoint", "structure"],
        "understand": ["overview", "architecture", "entrypoint"],
        "summarize": ["overview", "capabilities", "entrypoint"],
        "capabilities": ["capabilities", "entrypoint", "purpose"],
        "purpose": ["purpose", "capabilities", "entrypoint"],
        "components": ["components", "modules", "architecture"],
        "modules": ["components", "modules"],
        "entrypoint": ["entrypoint", "execution_flow"],
        "getting started": ["entrypoint", "overview", "architecture"],
        "onboarding": ["entrypoint", "overview", "architecture", "capabilities"],
    }

    # ---- Concept extraction regex ----
    _CONCEPT_PATTERN = re.compile(
        r'\b(?:'
        r'cli|ui|api|web|http|rest|graphql|database|db|sql|nosql|'
        r'auth|login|logout|session|token|jwt|oauth|'
        r'file|io|stream|network|socket|pipe|'
        r'config|settings|environment|env|'
        r'cache|queue|log|monitor|metric|'
        r'dispatch|route|command|handler|middleware|'
        r'parse|serialize|deserialize|format|convert|transform|'
        r'validate|sanitize|check|verify|'
        r'worker|job|task|scheduler|cron|'
        r'search|filter|sort|paginate|'
        r'upload|download|import|export|'
        r'crypto|encrypt|decrypt|hash|sign|'
        r'math|calc|compute|algorithm|'
        r'thread|async|parallel|concurrent|'
        r'state|store|persist|memory|'
        r'error|exception|retry|fallback|timeout'
        r')\b',
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskIntent:
        """Analyze task text and produce a TaskIntent with family classification.

        Two-layer classification:
        1. Determine IntentFamily (UNDERSTANDING, EXECUTION, NAVIGATION, ...)
        2. If EXECUTION: classify task type (bug_fix, feature_addition, ...)
           If UNDERSTANDING: classify subtype (overview, architecture, ...)
        """
        text = f"{task_title} {task_description} {goal}".lower()
        text_original = f"{task_title} {task_description} {goal}"

        # ---- Layer 1: Intent Family ----
        family, family_confidence = self._classify_family(text)

        # ---- Layer 2: Family-specific classification ----
        task_type = "unknown"
        type_confidence = 0.3

        if family == IntentFamily.EXECUTION:
            task_type, type_confidence = self._classify_task_type(text)

        elif family == IntentFamily.UNDERSTANDING:
            task_type, type_confidence = self._classify_understanding_subtype(text)

        elif family == IntentFamily.NAVIGATION:
            task_type = "navigation"
            type_confidence = family_confidence

        elif family == IntentFamily.EXPLANATION:
            task_type = "deep_dive"
            type_confidence = family_confidence

        elif family == IntentFamily.PLANNING:
            task_type = "planning"
            type_confidence = family_confidence

        # ---- Symbol extraction: ONLY for EXECUTION ----
        # For understanding/navigation/explanation queries, symbol extraction
        # is harmful — "understand" is not a code symbol.
        symbols: list[str] = []
        if family == IntentFamily.EXECUTION:
            symbols = self._extract_symbols(text_original)

        # ---- Concept extraction ----
        if family == IntentFamily.UNDERSTANDING:
            # Understanding queries: semantic mapping + regex concepts
            concepts = self._map_semantic_concepts(text)
            domain_concepts = self._extract_concepts(text)
            # Merge, semantic first
            seen = set(concepts)
            for c in domain_concepts:
                if c not in seen:
                    seen.add(c)
                    concepts.append(c)
        else:
            concepts = self._extract_concepts(text)

        # ---- Entrypoint relevance ----
        entrypoint_related = self._is_entrypoint_related(text, symbols)

        # ---- Affected files ----
        affected_files = self._guess_affected_files(text, symbols)

        return TaskIntent(
            family=family.value if hasattr(family, 'value') else str(family),
            type=task_type,
            symbols=symbols,
            concepts=concepts,
            entrypoint_related=entrypoint_related,
            affected_files=affected_files,
            confidence=max(type_confidence, family_confidence),
        )

    # ------------------------------------------------------------------
    # Layer 1: Intent Family classification
    # ------------------------------------------------------------------

    def _classify_family(self, text: str) -> tuple[IntentFamily, float]:
        """Classify the high-level intent family.

        Checks understanding/navigation/explanation/planning patterns first.
        If nothing matches, defaults to EXECUTION (the most common case).
        """
        scores: dict[IntentFamily, int] = {}

        # Check each family's patterns
        for pat in self._UNDERSTANDING_PATTERNS:
            if pat in text:
                scores[IntentFamily.UNDERSTANDING] = scores.get(IntentFamily.UNDERSTANDING, 0) + 1

        for pat in self._NAVIGATION_PATTERNS:
            if pat in text:
                scores[IntentFamily.NAVIGATION] = scores.get(IntentFamily.NAVIGATION, 0) + 1

        for pat in self._EXPLANATION_PATTERNS:
            if pat in text:
                scores[IntentFamily.EXPLANATION] = scores.get(IntentFamily.EXPLANATION, 0) + 1

        for pat in self._PLANNING_PATTERNS:
            if pat in text:
                scores[IntentFamily.PLANNING] = scores.get(IntentFamily.PLANNING, 0) + 1

        if not scores:
            # No non-execution patterns matched → default to EXECUTION
            return (IntentFamily.EXECUTION, 0.6)

        best_family = max(scores, key=lambda k: scores[k])
        best_score = scores[best_family]
        confidence = min(0.9, 0.5 + (best_score / max(1, sum(scores.values()))) * 0.4)

        return (best_family, confidence)

    # ------------------------------------------------------------------
    # Layer 2a: Task type classification (EXECUTION family)
    # ------------------------------------------------------------------

    def _classify_task_type(self, text: str) -> tuple[str, float]:
        """Classify execution task type by keyword pattern matching."""
        scores: dict[str, int] = {}
        for task_type, keywords in self._TYPE_PATTERNS:
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return ("unknown", 0.3)

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        confidence = min(0.9, 0.4 + (best_score / sum(scores.values())) * 0.5)
        return (best_type, confidence)

    # ------------------------------------------------------------------
    # Layer 2b: Understanding subtype classification
    # ------------------------------------------------------------------

    def _classify_understanding_subtype(self, text: str) -> tuple[str, float]:
        """Classify the type of understanding query."""
        scores: dict[str, int] = {}
        for subtype, patterns in self._UNDERSTANDING_SUBTYPES.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                scores[subtype] = score

        if not scores:
            return ("overview", 0.5)  # Default understanding subtype

        best = max(scores, key=lambda k: scores[k])
        confidence = min(0.9, 0.5 + (scores[best] / sum(scores.values())) * 0.4)
        return (best, confidence)

    # ------------------------------------------------------------------
    # Semantic concept mapping (for understanding queries)
    # ------------------------------------------------------------------

    def _map_semantic_concepts(self, text: str) -> list[str]:
        """Map natural-language understanding queries to architectural concepts.

        Handles Chinese, paraphrases, and open-ended questions by looking
        up patterns in the semantic concept map — not extracting keywords.
        """
        concepts: list[str] = []
        seen: set[str] = set()

        for pattern, mapped in self._SEMANTIC_CONCEPT_MAP.items():
            if pattern in text:
                for c in mapped:
                    if c not in seen:
                        seen.add(c)
                        concepts.append(c)

        # If nothing matched but it's clearly an understanding query
        # (e.g. "tell me about this codebase"), use default overview set
        if not concepts:
            concepts = ["architecture", "overview", "entrypoint", "capabilities"]

        return concepts[:8]

    # ------------------------------------------------------------------
    # Symbol extraction — ONLY for EXECUTION family
    # ------------------------------------------------------------------

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract potential code symbol names from task text.

        ONLY called for EXECUTION family queries.  Filters aggressively
        to avoid treating natural-language words as code symbols.
        """
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text)
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "should",
            "when", "will", "what", "which", "where", "have", "been", "does",
            "not", "are", "was", "can", "has", "had", "its", "all", "but",
            "just", "also", "into", "more", "some", "such", "than", "then",
            "now", "new", "use", "using", "used", "need", "needs", "make",
            "makes", "made", "add", "get", "set", "run", "see", "let",
            "implement", "create", "build", "write", "test", "fix", "change",
            "update", "remove", "delete", "ensure", "check", "verify",
            "task", "code", "file", "files", "function", "module", "class",
            "project", "goal", "description", "title", "step", "work",
            "want", "like", "would", "could", "must", "shall",
            # Task type keywords
            "integrate", "wire", "connect", "hook", "refactor", "rename",
            "install", "upgrade", "downgrade", "restructure", "reorganize",
            "introduce", "support",
            # Understanding keywords — NOT code symbols
            "understand", "overview", "summarize", "describe", "explain",
            "architecture", "component", "module", "feature", "capability",
            "purpose", "structure", "design", "pattern",
        }
        seen: set[str] = set()
        result: list[str] = []
        for s in identifiers:
            lower = s.lower()
            if lower not in stop_words and lower not in seen:
                seen.add(lower)
                result.append(s)
        return result[:8]

    # ------------------------------------------------------------------
    # Concept extraction (for execution/navigation queries)
    # ------------------------------------------------------------------

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract domain concepts from task text."""
        matches = self._CONCEPT_PATTERN.findall(text)
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            lower = m.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(lower)
        return result[:6]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_entrypoint_related(
        self, text: str, symbols: list[str]
    ) -> bool:
        entrypoint_keywords = {
            "main", "cli", "entry", "entrypoint", "entry_point",
            "command", "argparse", "click", "typer", "run", "start",
            "shell", "terminal", "console", "script",
        }
        text_lower = text.lower()
        if any(kw in text_lower for kw in entrypoint_keywords):
            return True
        if any("main" in s.lower() or "cli" in s.lower() for s in symbols):
            return True
        return False

    def _guess_affected_files(self, text: str, symbols: list[str]) -> list[str]:
        file_pattern = re.compile(r'\b[\w/\\-]+\.(?:py|yaml|yml|toml|json|md)\b')
        matches = file_pattern.findall(text)
        return list(dict.fromkeys(matches))[:5]
