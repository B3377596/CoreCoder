"""Task intent analysis — understand WHAT the user is trying to do.

Analyses task text to determine:
- Task type (bug_fix, feature_addition, cli_change, etc.)
- Target symbols mentioned
- Key concepts
- Whether this is entrypoint-related

This drives retrieval ranking: different task types prioritize different
kinds of files.  A cli_change should surface main.py and argparse code;
a bug_fix should surface error-prone files and test files.

Analysis is purely heuristic — no LLM calls.  The output feeds into
RetrievalQueryPlanner.
"""

from __future__ import annotations

import re
from corecoder.orchestration.retrieval.models import TaskIntent


class TaskIntentAnalyzer:
    """Heuristic task intent classifier.

    Usage:
        analyzer = TaskIntentAnalyzer()
        intent = analyzer.analyze("Integrate sqrt function into CLI")
        # → TaskIntent(type="feature_integration", symbols=["sqrt"], ...)
    """

    # Task type detection patterns — ordered by specificity
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

    # Concept detection — extract meaningful domain terms
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

    def analyze(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskIntent:
        """Analyze task text and produce a TaskIntent.

        Args:
            task_title: Short title of the task node.
            task_description: Detailed description.
            goal: Overall project goal.

        Returns:
            TaskIntent with type, symbols, concepts, and confidence.
        """
        text = f"{task_title} {task_description} {goal}".lower()
        text_original = f"{task_title} {task_description} {goal}"

        # 1. Determine task type
        task_type, type_confidence = self._classify_type(text)

        # 2. Extract symbol mentions (identifiers that look like code symbols)
        symbols = self._extract_symbols(text_original)

        # 3. Extract concepts
        concepts = self._extract_concepts(text)

        # 4. Check entrypoint relevance
        entrypoint_related = self._is_entrypoint_related(text, symbols)

        # 5. Guess affected files from task hints
        affected_files = self._guess_affected_files(text, symbols)

        return TaskIntent(
            type=task_type,
            symbols=symbols,
            concepts=concepts,
            entrypoint_related=entrypoint_related,
            affected_files=affected_files,
            confidence=type_confidence,
        )

    # ------------------------------------------------------------------
    # classification
    # ------------------------------------------------------------------

    def _classify_type(self, text: str) -> tuple[str, float]:
        """Classify task type by keyword pattern matching.

        Returns (type, confidence).  Order matters — first match wins,
        so more specific patterns are checked first.
        """
        scores: dict[str, int] = {}
        for task_type, keywords in self._TYPE_PATTERNS:
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return ("unknown", 0.3)

        # Pick the type with the most keyword matches
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        total = sum(scores.values())
        confidence = min(0.9, 0.4 + (best_score / total) * 0.5)

        return (best_type, confidence)

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract potential symbol names from task text.

        Looks for identifier-like tokens that might refer to code symbols.
        Filters out common English words and task-type keywords.
        """
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text)
        # Filter out common English words and known non-symbols
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
            # Task type keywords (not symbols)
            "integrate", "wire", "connect", "hook", "refactor", "rename",
            "install", "upgrade", "downgrade", "restructure", "reorganize",
            "introduce", "support",
        }
        # Deduplicate case-insensitively, preserving first occurrence
        seen: set[str] = set()
        result: list[str] = []
        for s in identifiers:
            lower = s.lower()
            if lower not in stop_words and lower not in seen:
                seen.add(lower)
                result.append(s)
        return result[:8]

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract domain concepts from task text."""
        matches = self._CONCEPT_PATTERN.findall(text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            lower = m.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(lower)
        return result[:6]

    def _is_entrypoint_related(
        self, text: str, symbols: list[str]
    ) -> bool:
        """Check if the task involves the application entry point."""
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
        """Try to guess which files are affected from task text.

        Looks for file path mentions like 'calculator.py' or 'src/main.py'.
        """
        file_pattern = re.compile(r'\b[\w/\\-]+\.(?:py|yaml|yml|toml|json|md)\b')
        matches = file_pattern.findall(text)
        return list(dict.fromkeys(matches))[:5]  # deduplicate, preserve order
