"""Task intent analysis for retrieval routing."""

from __future__ import annotations

import re

from corecoder.retrieval.models import IntentFamily, TaskIntent


class TaskIntentAnalyzer:
    """Classify task text into an intent family and subtype."""

    _UNDERSTANDING_PATTERNS: list[str] = [
        "what does this",
        "what is this",
        "how does this work",
        "explain the project",
        "explain the codebase",
        "explain the architecture",
        "project overview",
        "architecture overview",
        "understand",
        "overview",
        "summarize",
        "describe",
        "architecture",
        "structure",
        "onboarding",
        "getting started",
        "walk me through",
        "tell me about",
        "give me an overview",
    ]

    _NAVIGATION_PATTERNS: list[str] = [
        "where is",
        "find the",
        "locate",
        "show me",
        "which file",
        "file path",
        "where does",
    ]

    _EXPLANATION_PATTERNS: list[str] = [
        "how does",
        "why does",
        "how do",
        "why is",
        "explain how",
        "explain why",
        "deep dive",
    ]

    _PLANNING_PATTERNS: list[str] = [
        "what do i need",
        "how should i",
        "what's the best way",
        "plan",
        "approach",
        "strategy",
        "steps to",
    ]

    _TYPE_PATTERNS: list[tuple[str, list[str]]] = [
        ("bug_fix", ["fix", "bug", "error", "crash", "broken", "issue", "defect"]),
        ("dependency_change", ["install", "uninstall", "upgrade", "downgrade", "dependency", "package", "library"]),
        ("cli_change", ["cli", "command line", "argparse", "click", "typer", "flag", "option", "terminal"]),
        ("refactor", ["refactor", "clean up", "cleanup", "restructure", "reorganize", "extract", "split", "merge"]),
        ("rename", ["rename", "move to", "relocate", "change name"]),
        ("test_addition", ["test", "unittest", "pytest", "coverage", "assert", "mock", "fixture"]),
        ("feature_integration", ["integrate", "connect", "wire", "hook up", "combine", "plug into"]),
        ("feature_addition", ["add", "new", "create", "implement", "build", "introduce", "support for"]),
        ("documentation", ["document", "docstring", "comment", "readme", "describe"]),
    ]

    _UNDERSTANDING_SUBTYPES: dict[str, list[str]] = {
        "overview": ["overview", "what does this", "what is this", "summarize", "describe the project"],
        "architecture": ["architecture", "how is this organized", "how is the code", "design", "pattern"],
        "capabilities": ["capabilities", "features", "what can", "functionality", "purpose"],
        "components": ["modules", "components", "packages", "directories"],
    }

    _SEMANTIC_CONCEPT_MAP: dict[str, list[str]] = {
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

    _CONCEPT_PATTERN = re.compile(
        r"\b("
        r"cli|ui|api|web|http|rest|graphql|database|db|sql|nosql|"
        r"auth|login|logout|session|token|jwt|oauth|"
        r"file|io|stream|network|socket|pipe|"
        r"config|settings|environment|env|"
        r"cache|queue|log|monitor|metric|"
        r"dispatch|route|command|handler|middleware|"
        r"parse|serialize|deserialize|format|convert|transform|"
        r"validate|sanitize|check|verify|"
        r"worker|job|task|scheduler|cron|"
        r"search|filter|sort|paginate|"
        r"upload|download|import|export|"
        r"crypto|encrypt|decrypt|hash|sign|"
        r"math|calc|compute|algorithm|"
        r"thread|async|parallel|concurrent|"
        r"state|store|persist|memory|"
        r"error|exception|retry|fallback|timeout|"
        r"compression|compress|retrieval|planner|query|index|indexing|"
        r"prompt|tool|git|shadow|benchmark|orchestration"
        r")\b",
        re.IGNORECASE,
    )

    def analyze(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskIntent:
        text = f"{task_title} {task_description} {goal}".lower()
        text_original = f"{task_title} {task_description} {goal}"

        family, family_confidence = self._classify_family(text)

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

        symbols: list[str] = []
        if family in (IntentFamily.EXECUTION, IntentFamily.NAVIGATION, IntentFamily.EXPLANATION):
            symbols = self._extract_symbols(text_original)

        if family == IntentFamily.UNDERSTANDING:
            concepts = self._map_semantic_concepts(text)
            for concept in self._extract_concepts(text):
                if concept not in concepts:
                    concepts.append(concept)
        else:
            concepts = self._extract_concepts(text)

        return TaskIntent(
            family=family.value if hasattr(family, "value") else str(family),
            type=task_type,
            symbols=symbols,
            concepts=concepts[:8],
            entrypoint_related=self._is_entrypoint_related(text, symbols),
            affected_files=self._guess_affected_files(text, symbols),
            confidence=max(type_confidence, family_confidence),
        )

    def _classify_family(self, text: str) -> tuple[IntentFamily, float]:
        scores: dict[IntentFamily, int] = {}
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
            return IntentFamily.EXECUTION, 0.6
        best = max(scores, key=lambda family: scores[family])
        confidence = min(0.9, 0.5 + (scores[best] / max(1, sum(scores.values()))) * 0.4)
        return best, confidence

    def _classify_task_type(self, text: str) -> tuple[str, float]:
        scores: dict[str, int] = {}
        for task_type, keywords in self._TYPE_PATTERNS:
            score = sum(1 for keyword in keywords if keyword in text)
            if score:
                scores[task_type] = score
        if not scores:
            return "unknown", 0.3
        best = max(scores, key=lambda task_type: scores[task_type])
        confidence = min(0.9, 0.4 + (scores[best] / sum(scores.values())) * 0.5)
        return best, confidence

    def _classify_understanding_subtype(self, text: str) -> tuple[str, float]:
        scores: dict[str, int] = {}
        for subtype, patterns in self._UNDERSTANDING_SUBTYPES.items():
            score = sum(1 for pattern in patterns if pattern in text)
            if score:
                scores[subtype] = score
        if not scores:
            return "overview", 0.5
        best = max(scores, key=lambda subtype: scores[subtype])
        confidence = min(0.9, 0.5 + (scores[best] / sum(scores.values())) * 0.4)
        return best, confidence

    def _map_semantic_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        for pattern, mapped in self._SEMANTIC_CONCEPT_MAP.items():
            if pattern in text:
                for concept in mapped:
                    if concept not in concepts:
                        concepts.append(concept)
        return concepts or ["architecture", "overview", "entrypoint", "capabilities"]

    def _extract_symbols(self, text: str) -> list[str]:
        identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text)
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "should", "when", "will",
            "what", "which", "where", "have", "been", "does", "not", "are", "was", "can",
            "has", "had", "its", "all", "but", "just", "also", "into", "more", "some",
            "such", "than", "then", "now", "new", "use", "using", "used", "need", "needs",
            "make", "makes", "made", "add", "get", "set", "run", "see", "let", "implement",
            "create", "build", "write", "test", "fix", "change", "update", "remove", "delete",
            "ensure", "check", "verify", "task", "code", "file", "files", "function", "module",
            "class", "project", "goal", "description", "title", "step", "work", "want", "like",
            "would", "could", "must", "shall", "integrate", "wire", "connect", "hook", "refactor",
            "rename", "install", "upgrade", "downgrade", "restructure", "reorganize", "introduce",
            "support", "understand", "overview", "summarize", "describe", "explain", "architecture",
            "component", "feature", "capability", "purpose", "structure", "design", "pattern",
            "implemented", "implementation", "logic", "works", "system", "layer",
        }
        result: list[str] = []
        seen: set[str] = set()
        for symbol in identifiers:
            lowered = symbol.lower()
            if lowered not in stop_words and lowered not in seen:
                seen.add(lowered)
                result.append(symbol)
        return result[:8]

    def _extract_concepts(self, text: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for match in self._CONCEPT_PATTERN.findall(text):
            lowered = match.lower()
            if lowered not in seen:
                seen.add(lowered)
                result.append(lowered)
        return result[:6]

    def _is_entrypoint_related(self, text: str, symbols: list[str]) -> bool:
        entrypoint_keywords = {
            "main", "cli", "entry", "entrypoint", "entry_point", "command",
            "argparse", "click", "typer", "run", "start", "shell", "terminal",
            "console", "script",
        }
        if any(keyword in text for keyword in entrypoint_keywords):
            return True
        return any("main" in symbol.lower() or "cli" in symbol.lower() for symbol in symbols)

    def _guess_affected_files(self, text: str, symbols: list[str]) -> list[str]:
        file_pattern = re.compile(r"\b[\w/\\.-]+\.(?:py|yaml|yml|toml|json|md)\b")
        hints: list[str] = list(dict.fromkeys(file_pattern.findall(text)))
        seen = {hint.lower() for hint in hints}

        phrase_patterns = [
            "query planner",
            "context orchestrator",
            "dependency graph",
            "symbol index",
            "repo info",
            "repo_info",
            "shadow git",
            "runtime state",
            "task intent",
            "file summaries",
            "session persistence",
            "context compression",
            "repository indexing",
        ]
        for phrase in phrase_patterns:
            if phrase in text:
                candidate = f"{phrase.replace(' ', '_')}.py"
                if candidate.lower() not in seen:
                    seen.add(candidate.lower())
                    hints.append(candidate)

        for symbol in symbols:
            lowered = symbol.lower()
            if "_" in lowered:
                candidate = f"{lowered}.py"
                if candidate not in seen:
                    seen.add(candidate)
                    hints.append(candidate)

        return hints[:8]
