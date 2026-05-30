"""Task understanding and legacy intent compatibility for Retrieval V2."""

from __future__ import annotations

import re

from corecoder.retrieval.models import (
    IntentFamily,
    TaskConstraint,
    TaskEntity,
    TaskIntent,
    TaskUnderstanding,
)


class TaskIntentAnalyzer:
    """Produces rich task understanding, plus a compatibility TaskIntent view.

    Retrieval V2 weakens hard-coded task classification.  Instead of mapping
    every request into narrow labels like ``bug_fix`` or ``feature_addition``,
    this analyzer extracts:

    - goal/objective
    - entities (symbols/modules/files)
    - constraints
    - likely modules / scopes

    ``analyze()`` is retained as a compatibility wrapper because the rest of the
    retrieval stack still expects a ``TaskIntent`` during migration.
    """

    _UNDERSTANDING_HINTS = (
        "what does this",
        "what is this",
        "overview",
        "architecture",
        "structure",
        "onboarding",
        "walk me through",
        "give me an overview",
    )
    _NAVIGATION_HINTS = (
        "where is",
        "which file",
        "locate",
        "show me",
        "path",
    )
    _EXPLANATION_HINTS = (
        "how does",
        "why does",
        "how do",
        "why is",
        "deep dive",
    )
    _PLANNING_HINTS = (
        "how should i",
        "what do i need",
        "plan",
        "strategy",
        "steps to",
    )
    _CONSTRAINT_PATTERNS = (
        r"\bwithout\s+([a-zA-Z0-9_\- ]+)",
        r"\bdo not\s+([a-zA-Z0-9_\- ]+)",
        r"\bmust\s+([a-zA-Z0-9_\- ]+)",
        r"\bonly\s+([a-zA-Z0-9_\- ]+)",
    )
    _MODULE_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:/[a-zA-Z0-9_\-]+)*(?:\.[a-zA-Z0-9_]+)?\b")
    _FILE_PATTERN = re.compile(r"\b[\w/\\.-]+\.(?:py|yaml|yml|toml|json|md)\b")
    _CAMEL_CASE_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*\b")
    _SNAKE_CASE_PATTERN = re.compile(r"\b[a-z_][a-z0-9_]{2,}\b")
    _QUERY_TERM_PATTERN = re.compile(
        r"\b(auth|login|session|token|config|cli|prompt|context|planner|retrieval|"
        r"index|graph|dependency|scheduler|verifier|runtime|state|memory|tool|mcp|"
        r"compression|history|shadow|repo|symbol)\b",
        re.IGNORECASE,
    )
    _STOP_WORDS = {
        "what", "where", "when", "which", "with", "without", "should", "does",
        "this", "that", "these", "those", "have", "been", "being", "into",
        "from", "show", "tell", "about", "give", "need", "must", "only",
        "work", "works", "working", "implemented", "implementation", "logic",
        "module", "modules", "file", "files", "code", "project", "system",
        "layer", "build", "change", "modify", "update", "create", "write",
        "fix", "understand", "explain", "plan", "strategy", "steps",
    }

    def understand(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskUnderstanding:
        text_original = " ".join(part for part in (task_title, task_description, goal) if part).strip()
        text = text_original.lower()

        objective = goal or task_description or task_title or text_original
        entities = self._extract_entities(text_original)
        constraints = self._extract_constraints(text)
        likely_modules = self._guess_likely_modules(text, entities)
        query_terms = self._extract_query_terms(text)
        confidence = self._estimate_confidence(entities, likely_modules, query_terms)

        return TaskUnderstanding(
            goal=text_original,
            objective=objective.strip(),
            entities=entities,
            constraints=constraints,
            likely_modules=likely_modules,
            query_terms=query_terms,
            confidence=confidence,
        )

    def analyze(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskIntent:
        understanding = self.understand(task_title, task_description, goal)
        family = self._infer_family(understanding)
        symbols = [e.name for e in understanding.entities if e.kind in {"symbol", "class", "function", "method"}]
        concepts = list(dict.fromkeys(understanding.query_terms + understanding.likely_modules))[:8]
        affected_files = self._guess_affected_files(understanding)

        return TaskIntent(
            family=family.value,
            type=self._infer_legacy_type(family, understanding),
            symbols=symbols[:8],
            concepts=concepts,
            entrypoint_related=any(term in ("cli", "command", "entrypoint") for term in understanding.query_terms),
            affected_files=affected_files,
            confidence=understanding.confidence,
            understanding=understanding,
        )

    def _infer_family(self, understanding: TaskUnderstanding) -> IntentFamily:
        text = understanding.goal.lower()
        if any(hint in text for hint in self._NAVIGATION_HINTS):
            return IntentFamily.NAVIGATION
        if any(hint in text for hint in self._EXPLANATION_HINTS):
            return IntentFamily.EXPLANATION
        if any(hint in text for hint in self._UNDERSTANDING_HINTS):
            return IntentFamily.UNDERSTANDING
        if any(hint in text for hint in self._PLANNING_HINTS):
            return IntentFamily.PLANNING
        return IntentFamily.EXECUTION

    def _infer_legacy_type(self, family: IntentFamily, understanding: TaskUnderstanding) -> str:
        if family == IntentFamily.NAVIGATION:
            return "navigation"
        if family == IntentFamily.EXPLANATION:
            return "deep_dive"
        if family == IntentFamily.PLANNING:
            return "planning"
        if family == IntentFamily.UNDERSTANDING:
            if "architecture" in understanding.query_terms:
                return "architecture"
            return "overview"
        return "general_task"

    def _extract_entities(self, text_original: str) -> list[TaskEntity]:
        entities: list[TaskEntity] = []
        seen: set[tuple[str, str]] = set()

        def add(name: str, kind: str, confidence: float, source: str) -> None:
            key = (name, kind)
            if not name or key in seen:
                return
            seen.add(key)
            entities.append(TaskEntity(name=name, kind=kind, confidence=confidence, source=source))

        for match in self._FILE_PATTERN.findall(text_original):
            add(match.replace("\\", "/"), "file", 0.95, "file_pattern")

        for match in self._CAMEL_CASE_PATTERN.findall(text_original):
            add(match, "class", 0.9, "camel_case")

        for match in self._SNAKE_CASE_PATTERN.findall(text_original):
            lowered = match.lower()
            if lowered in self._STOP_WORDS:
                continue
            kind = "symbol" if "_" in match else "module"
            add(match, kind, 0.65 if kind == "symbol" else 0.55, "snake_case")

        return entities[:12]

    def _extract_constraints(self, text: str) -> list[TaskConstraint]:
        constraints: list[TaskConstraint] = []
        seen: set[str] = set()
        for pattern in self._CONSTRAINT_PATTERNS:
            for match in re.findall(pattern, text):
                normalized = match.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    constraints.append(TaskConstraint(text=normalized))
        return constraints[:6]

    def _guess_likely_modules(self, text: str, entities: list[TaskEntity]) -> list[str]:
        modules: list[str] = []
        seen: set[str] = set()

        for match in self._QUERY_TERM_PATTERN.findall(text):
            lowered = match.lower()
            if lowered not in seen:
                seen.add(lowered)
                modules.append(lowered)

        for entity in entities:
            if entity.kind == "file":
                stem = entity.name.split("/")[-1].rsplit(".", 1)[0]
                if stem not in seen:
                    seen.add(stem)
                    modules.append(stem)
            elif entity.kind == "module":
                lowered = entity.name.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    modules.append(lowered)

        return modules[:10]

    def _extract_query_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for match in self._QUERY_TERM_PATTERN.findall(text):
            lowered = match.lower()
            if lowered not in seen:
                seen.add(lowered)
                terms.append(lowered)
        return terms[:10]

    def _estimate_confidence(
        self,
        entities: list[TaskEntity],
        likely_modules: list[str],
        query_terms: list[str],
    ) -> float:
        score = 0.35
        score += min(0.25, len(entities) * 0.05)
        score += min(0.2, len(likely_modules) * 0.03)
        score += min(0.2, len(query_terms) * 0.03)
        return min(0.95, score)

    def _guess_affected_files(self, understanding: TaskUnderstanding) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()

        for entity in understanding.entities:
            if entity.kind == "file":
                normalized = entity.name.replace("\\", "/")
                if normalized not in seen:
                    seen.add(normalized)
                    hints.append(normalized)
                continue

            if entity.kind in {"symbol", "module"}:
                candidate = f"{entity.name.lower()}.py"
                if candidate not in seen:
                    seen.add(candidate)
                    hints.append(candidate)

        phrase_map = {
            "query planner": "query_planner.py",
            "context orchestrator": "orchestrator.py",
            "context compression": "compression.py",
            "repository indexing": "index.py",
            "dependency graph": "dependency_graph.py",
            "task intent": "task_intent.py",
            "repo info": "repo_info.py",
            "runtime state": "state.py",
            "shadow git": "shadow.py",
        }
        lowered_goal = understanding.goal.lower()
        for phrase, filename in phrase_map.items():
            if phrase in lowered_goal and filename not in seen:
                seen.add(filename)
                hints.append(filename)

        return hints[:10]
