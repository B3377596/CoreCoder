"""Task understanding and retrieval-hint inference for Retrieval V2."""

from __future__ import annotations

import re

from corecoder.retrieval.models import (
    IntentFamily,
    TaskConstraint,
    TaskEntity,
    TaskUnderstanding,
)


class TaskUnderstandingAnalyzer:
    """Extract task semantics and project retrieval-facing hints.

    This module is no longer an "intent classifier" in the old sense.
    Its job is to:
    - build a richer TaskUnderstanding object
    - infer lightweight retrieval-routing hints
    - project code-oriented query hints for planner/ranker use
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
        "这个项目在做什么",
        "项目在做什么",
        "项目概览",
        "项目架构",
        "整体架构",
        "整体结构",
        "项目结构",
        "带我了解",
    )
    _NAVIGATION_HINTS = (
        "where is",
        "which file",
        "locate",
        "show me",
        "path",
        "在哪里",
        "在哪",
        "哪个文件",
        "文件在哪",
        "定位",
    )
    _EXPLANATION_HINTS = (
        "how does",
        "why does",
        "how do",
        "why is",
        "deep dive",
        "怎么实现",
        "如何实现",
        "怎么工作",
        "如何工作",
        "怎么做的",
        "解释一下",
        "讲一下",
        "原理",
    )
    _PLANNING_HINTS = (
        "how should i",
        "what do i need",
        "plan",
        "strategy",
        "steps to",
        "怎么改",
        "如何改",
        "怎么做",
        "如何做",
        "方案",
        "计划",
        "步骤",
    )
    _CONSTRAINT_PATTERNS = (
        r"\bwithout\s+([a-zA-Z0-9_\- ]+)",
        r"\bdo not\s+([a-zA-Z0-9_\- ]+)",
        r"\bmust\s+([a-zA-Z0-9_\- ]+)",
        r"\bonly\s+([a-zA-Z0-9_\- ]+)",
        r"不要([^\s，。,;；]+)",
        r"不能([^\s，。,;；]+)",
        r"必须([^\s，。,;；]+)",
        r"只([^\s，。,;；]+)",
    )
    _FILE_PATTERN = re.compile(r"\b[\w/\\.-]+\.(?:py|yaml|yml|toml|json|md)\b")
    _CAMEL_CASE_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*\b")
    _SNAKE_CASE_PATTERN = re.compile(r"\b[a-z_][a-z0-9_]{2,}\b")
    _ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
    _STOP_WORDS = {
        "what", "where", "when", "which", "with", "without", "should", "does",
        "how", "why", "who", "whom", "whose",
        "this", "that", "these", "those", "have", "been", "being", "into",
        "from", "show", "tell", "about", "give", "need", "must", "only",
        "the", "for", "and", "are", "is", "was", "were", "be", "been", "being",
        "get", "got", "do", "did", "done", "can", "could", "would", "will",
        "or", "of", "to", "in", "on", "by", "at", "as", "if", "than", "then",
        "before", "after", "during", "through", "across", "like",
        "work", "works", "working", "implemented", "implementation", "logic",
        "module", "modules", "file", "files", "code", "project", "system",
        "layer", "build", "change", "modify", "update", "create", "write",
        "fix", "understand", "explain", "plan", "strategy", "steps",
        "request", "requests", "response", "responses", "defined", "generated",
        "supported", "implemented", "stored", "modeled", "derived", "ranked",
        "selection", "common", "main", "local", "first", "more", "next",
    }
    _STOP_WORDS_ZH = {
        "这个", "那个", "这个项目", "该项目", "项目", "一下", "请问", "帮我",
        "怎么", "如何", "为什么", "哪里", "哪个", "哪些", "一下子", "一下吧",
        "实现", "逻辑", "功能", "代码", "模块", "文件", "系统", "项目里",
        "项目中", "工作", "作用", "原理", "机制", "流程", "相关", "里面",
    }
    _QUESTION_WORDS = {"how", "what", "where", "why", "which", "who", "when"}
    _NOISE_MODULE_WORDS = {
        "the", "for", "and", "are", "is", "was", "were", "be", "been", "being",
        "do", "does", "did", "done", "get", "got", "into", "from", "with",
        "without", "before", "after", "during", "through", "across", "like",
        "need", "must", "only", "about", "show", "tell", "give", "work", "works",
        "working", "implemented", "implementation", "supported", "generated",
        "defined", "stored", "modeled", "derived", "ranked", "selection",
        "common", "main", "local", "first", "more", "next", "or", "of", "to",
        "in", "on", "by", "at", "as", "if", "than", "then",
    }
    _CANONICAL_TERMS = {
        "auth": ("auth", "authentication", "认证", "鉴权", "授权"),
        "login": ("login", "登录"),
        "session": ("session", "会话"),
        "token": ("token", "令牌"),
        "config": ("config", "configuration", "配置"),
        "cli": ("cli", "command line", "命令行"),
        "prompt": ("prompt", "提示词"),
        "context": ("context", "上下文"),
        "planner": ("planner", "规划", "计划器", "规划器"),
        "retrieval": ("retrieval", "检索", "召回"),
        "index": ("index", "indexing", "索引", "索引构建"),
        "graph": ("graph", "图"),
        "dependency": ("dependency", "dependencies", "依赖"),
        "scheduler": ("scheduler", "调度", "调度器"),
        "verifier": ("verifier", "验证", "校验", "校验器"),
        "runtime": ("runtime", "运行时"),
        "state": ("state", "状态"),
        "memory": ("memory", "记忆"),
        "tool": ("tool", "tools", "工具"),
        "mcp": ("mcp",),
        "compression": ("compression", "压缩"),
        "history": ("history", "历史"),
        "shadow": ("shadow", "影子"),
        "repo": ("repo", "repository", "仓库"),
        "symbol": ("symbol", "symbols", "符号"),
        "workflow": ("workflow", "工作流"),
        "orchestrator": ("orchestrator", "编排", "编排器"),
        "dag": ("dag",),
        "read": ("read", "读取"),
        "write": ("write", "写入"),
        "grep": ("grep", "搜索"),
        "glob": ("glob", "globbing"),
    }
    _FALLBACK_THRESHOLD = 0.45

    def understand(
        self,
        task_title: str = "",
        task_description: str = "",
        goal: str = "",
    ) -> TaskUnderstanding:
        text_original = self._compose_text(task_title, task_description, goal)
        text = self._normalize_text(text_original)
        canonical_text = self._canonicalize_text(text)

        objective = self._compose_text(goal, task_description, task_title) or text_original
        entities = self._extract_entities(text_original)
        constraints = self._extract_constraints(text)
        likely_modules = self._guess_likely_modules(canonical_text, entities)
        query_terms = self._extract_query_terms(canonical_text)
        confidence = self._estimate_confidence(entities, likely_modules, query_terms)

        understanding = TaskUnderstanding(
            goal=text_original,
            objective=objective.strip(),
            entities=entities,
            constraints=constraints,
            likely_modules=likely_modules,
            query_terms=query_terms,
            confidence=confidence,
        )
        if self._needs_fallback(understanding):
            understanding = self._apply_low_confidence_fallback(understanding, text_original, canonical_text)
        return understanding

    def infer_retrieval_family(self, understanding: TaskUnderstanding) -> IntentFamily:
        text = self._canonicalize_text(self._normalize_text(understanding.goal))
        if any(hint in text for hint in self._NAVIGATION_HINTS):
            return IntentFamily.NAVIGATION
        if any(hint in text for hint in self._EXPLANATION_HINTS):
            return IntentFamily.EXPLANATION
        if any(hint in text for hint in self._UNDERSTANDING_HINTS):
            return IntentFamily.UNDERSTANDING
        if any(hint in text for hint in self._PLANNING_HINTS):
            return IntentFamily.PLANNING
        return IntentFamily.EXECUTION

    def infer_retrieval_task_type(self, understanding: TaskUnderstanding) -> str:
        family = self.infer_retrieval_family(understanding)
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

    def build_retrieval_hints(self, understanding: TaskUnderstanding) -> tuple[list[str], list[str], list[str]]:
        symbols = [
            entity.name
            for entity in understanding.entities
            if entity.kind in {"symbol", "class", "function", "method"}
        ]
        concepts = list(dict.fromkeys(understanding.query_terms + understanding.likely_modules))[:8]
        likely_files = self._guess_affected_files(understanding)
        return symbols[:8], concepts, likely_files

    # ------------------------------------------------------------------
    # Legacy compatibility wrappers
    # ------------------------------------------------------------------

    def infer_family(self, understanding: TaskUnderstanding) -> IntentFamily:
        """Backward-compatible wrapper for older retrieval code."""
        return self.infer_retrieval_family(understanding)

    def infer_task_type(self, understanding: TaskUnderstanding) -> str:
        """Backward-compatible wrapper for older retrieval code."""
        return self.infer_retrieval_task_type(understanding)

    def build_query_hints(self, understanding: TaskUnderstanding) -> tuple[list[str], list[str], list[str]]:
        """Backward-compatible wrapper for older retrieval code."""
        return self.build_retrieval_hints(understanding)

    def _extract_entities(self, text_original: str) -> list[TaskEntity]:
        entities: list[TaskEntity] = []
        seen: set[tuple[str, str]] = set()
        normalized_text = self._normalize_text(text_original)
        canonical_text = self._canonicalize_text(normalized_text)

        def add(name: str, kind: str, confidence: float, source: str) -> None:
            key = (name, kind)
            if not name or key in seen:
                return
            seen.add(key)
            entities.append(TaskEntity(name=name, kind=kind, confidence=confidence, source=source))

        for match in self._FILE_PATTERN.findall(text_original):
            add(match.replace("\\", "/"), "file", 0.95, "file_pattern")

        for match in self._CAMEL_CASE_PATTERN.findall(text_original):
            if self._is_noise_camel_case(match):
                continue
            add(match, "class", 0.9, "camel_case")

        for match in self._ACRONYM_PATTERN.findall(text_original):
            if match.lower() in self._STOP_WORDS:
                continue
            add(match, "class", 0.85, "acronym")

        for match in self._SNAKE_CASE_PATTERN.findall(text_original):
            lowered = match.lower()
            if lowered in self._STOP_WORDS or lowered in self._NOISE_MODULE_WORDS:
                continue
            kind = "symbol" if "_" in match else "module"
            add(match, kind, 0.65 if kind == "symbol" else 0.55, "snake_case")

        for canonical in self._extract_query_terms(canonical_text):
            add(canonical, "module", 0.6, "canonical_term")

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

        for match in self._extract_query_terms(text):
            if match not in seen:
                seen.add(match)
                modules.append(match)

        for entity in entities:
            if entity.kind == "file":
                stem = entity.name.split("/")[-1].rsplit(".", 1)[0]
                if stem not in seen:
                    seen.add(stem)
                    modules.append(stem)
            elif entity.kind == "module":
                lowered = entity.name.lower()
                if lowered not in seen and lowered not in self._NOISE_MODULE_WORDS:
                    seen.add(lowered)
                    modules.append(lowered)

        return modules[:10]

    def _extract_query_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        lowered = text.lower()
        for canonical, aliases in self._CANONICAL_TERMS.items():
            for alias in aliases:
                if alias in lowered:
                    if canonical not in seen:
                        seen.add(canonical)
                        terms.append(canonical)
                    break
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
            "检索规划": "query_planner.py",
            "context orchestrator": "orchestrator.py",
            "上下文编排": "orchestrator.py",
            "context compression": "compression.py",
            "上下文压缩": "compression.py",
            "repository indexing": "index.py",
            "仓库索引": "index.py",
            "dependency graph": "dependency_graph.py",
            "依赖图": "dependency_graph.py",
            "task understanding": "task_understanding.py",
            "task intent": "task_understanding.py",
            "任务理解": "task_understanding.py",
            "repo info": "repo_info.py",
            "仓库信息": "repo_info.py",
            "runtime state": "state.py",
            "运行时状态": "state.py",
            "shadow git": "shadow.py",
        }
        lowered_goal = self._canonicalize_text(self._normalize_text(understanding.goal))
        for phrase, filename in phrase_map.items():
            if phrase in lowered_goal and filename not in seen:
                seen.add(filename)
                hints.append(filename)

        return hints[:10]

    def _compose_text(self, *parts: str) -> str:
        unique_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = self._normalize_text(part)
            if not normalized:
                continue
            dedupe_key = normalized.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            unique_parts.append(normalized)
        return " ".join(unique_parts).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = (
            text.replace("，", " ")
            .replace("。", " ")
            .replace("：", " ")
            .replace("；", " ")
            .replace("（", " ")
            .replace("）", " ")
            .replace("、", " ")
            .replace("？", " ")
            .replace("！", " ")
            .replace("-", " ")
        )
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _canonicalize_text(self, text: str) -> str:
        lowered = text.lower()
        pieces = [lowered]
        for canonical, aliases in self._CANONICAL_TERMS.items():
            if any(alias in lowered for alias in aliases):
                pieces.append(canonical)
        return " ".join(dict.fromkeys(piece for piece in pieces if piece)).strip()

    def _needs_fallback(self, understanding: TaskUnderstanding) -> bool:
        if understanding.confidence < self._FALLBACK_THRESHOLD:
            return True
        return not understanding.entities and not understanding.query_terms

    def _apply_low_confidence_fallback(
        self,
        understanding: TaskUnderstanding,
        text_original: str,
        canonical_text: str,
    ) -> TaskUnderstanding:
        entities = list(understanding.entities)
        seen_entities = {(entity.name, entity.kind) for entity in entities}

        for canonical in self._extract_query_terms(canonical_text):
            key = (canonical, "module")
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append(TaskEntity(name=canonical, kind="module", confidence=0.5, source="fallback_term"))

        likely_modules = list(dict.fromkeys(understanding.likely_modules + self._extract_query_terms(canonical_text)))[:10]
        query_terms = list(dict.fromkeys(understanding.query_terms + self._extract_query_terms(canonical_text)))[:10]
        confidence = self._estimate_confidence(entities, likely_modules, query_terms)
        confidence = min(0.8, confidence + 0.05) if query_terms else confidence

        return TaskUnderstanding(
            goal=understanding.goal,
            objective=understanding.objective,
            entities=entities[:12],
            constraints=understanding.constraints,
            likely_modules=likely_modules,
            query_terms=query_terms,
            confidence=confidence,
        )

    def _is_noise_camel_case(self, value: str) -> bool:
        lowered = value.lower()
        if lowered in self._STOP_WORDS or lowered in self._QUESTION_WORDS:
            return True
        if len(value) <= 3 and value.upper() != value:
            return True
        # Treat a single leading-capital word like "How" or "Where" as noise
        # unless it looks like a real acronym or multi-hump code symbol.
        if value[0].isupper() and value[1:].islower():
            return True
        return False
