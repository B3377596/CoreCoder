"""Heuristic file summarization — no embeddings, no LLM calls.

Generates FileSummary objects from structural code analysis:
- Symbol names → infer purpose
- File path → infer category
- Import patterns → infer role
- Module structure → infer responsibilities

All summaries are cached in .corecoder/file_summaries.json.
Each summary is ~50-100 tokens — designed for agent reasoning loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corecoder.orchestration.retrieval.models import FileSummary


class FileSummaryManager:
    """Heuristic file summarizer.

    Categorizes files by analyzing their path, symbol names, and import
    patterns.  No LLM, no embeddings — pure structural heuristics.

    Usage:
        mgr = FileSummaryManager(working_dir="/path/to/repo")
        mgr.build(symbols_json)
        summary = mgr.get("main.py")
        matches = mgr.search_concept("CLI dispatch")
    """

    def __init__(self, working_dir: str = "."):
        self._working_dir = Path(working_dir)
        self._summaries: dict[str, FileSummary] = {}
        self._cache_path = self._working_dir / ".corecoder" / "file_summaries.json"

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, symbols_json: dict[str, Any]) -> None:
        """Build summaries for all files in the symbol index."""
        self._summaries.clear()

        for filepath, symbols in symbols_json.items():
            if self._should_skip(filepath):
                continue
            filepath = filepath.replace("\\", "/")

            symbol_names = self._extract_names(symbols)
            category = self._categorize(filepath, symbol_names)
            purpose = self._infer_purpose(filepath, symbol_names, category)
            responsibilities = self._infer_responsibilities(
                filepath, symbol_names, category
            )

            self._summaries[filepath] = FileSummary(
                path=filepath,
                purpose=purpose,
                responsibilities=responsibilities,
                key_symbols=self._key_symbols(symbol_names, category),
                category=category,
                file_type=self._file_type(filepath),
            )

    # ------------------------------------------------------------------
    # LLM-powered summary generation (optional, async)
    # ------------------------------------------------------------------

    async def generate_with_llm(
        self,
        filepath: str,
        symbols: list[str],
        llm_call,
        source_lines: str = "",
    ) -> FileSummary:
        """Generate a file summary using an LLM (async, one file).

        Uses a minimal prompt (~100 input tokens) to produce a structured
        summary.  Designed for offline/batch use — call once per file,
        cache the result.

        Args:
            filepath: Path to the file.
            symbols: List of symbol names in the file.
            llm_call: Async callable that takes messages and returns a
                      response with a .content attribute.
            source_lines: Optional first ~20 lines of the file for context.

        Returns:
            FileSummary with LLM-generated purpose and responsibilities.
        """
        symbol_list = ", ".join(symbols[:15]) if symbols else "(none)"
        fname = filepath.split("/")[-1]
        context = source_lines[:800] if source_lines else ""

        prompt = f"""Analyze this Python file and return a one-sentence purpose and 1-3 key responsibilities.

File: {fname}
Full path: {filepath}
Symbols defined: {symbol_list}
First lines:
{context}

Return ONLY valid JSON:
{{
  "purpose": "one short sentence describing what this file does",
  "responsibilities": ["short phrase 1", "short phrase 2"],
  "category": "cli|core_logic|utility|config|test|data|web"
}}"""

        try:
            resp = await llm_call([{"role": "user", "content": prompt}])
            import json as _json
            data = _json.loads(resp.content.strip())
            purpose = data.get("purpose", self._infer_purpose(filepath, symbols, "core_logic"))
            responsibilities = data.get("responsibilities", [])[:3]
            category = data.get("category", self._categorize(filepath, symbols))

            summary = FileSummary(
                path=filepath,
                purpose=purpose,
                responsibilities=responsibilities,
                key_symbols=self._key_symbols(symbols, category),
                category=category,
                file_type=self._file_type(filepath),
            )
            self._summaries[filepath] = summary
            return summary
        except Exception:
            # LLM call failed — fall back to heuristic
            return self._summaries.get(filepath) or FileSummary(path=filepath)

    async def generate_all_with_llm(
        self,
        symbols_json: dict,
        llm_call,
    ) -> None:
        """Generate LLM summaries for all files (async batch).

        Each file gets its own LLM call.  Results replace heuristic
        summaries and are cached to disk.
        """
        import asyncio

        async def _one(filepath: str, syms):
            names = self._extract_names(syms)
            # Try to read first lines for context
            source = ""
            full_path = self._working_dir / filepath
            if full_path.exists():
                try:
                    source = full_path.read_text(encoding="utf-8", errors="replace")[:800]
                except Exception:
                    pass
            await self.generate_with_llm(filepath, names, llm_call, source)

        tasks = []
        for fp, syms in symbols_json.items():
            if self._should_skip(fp):
                continue
            tasks.append(_one(fp.replace("\\", "/"), syms))

        # Run in parallel batches of 5 to avoid rate limiting
        for i in range(0, len(tasks), 5):
            batch = tasks[i:i + 5]
            await asyncio.gather(*batch)

        self.save_cache()

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def get(self, filepath: str) -> FileSummary | None:
        """Get the summary for a specific file."""
        filepath = filepath.replace("\\", "/")
        return self._summaries.get(filepath)

    def all_summaries(self) -> dict[str, FileSummary]:
        return dict(self._summaries)

    def search_concept(self, concept: str) -> list[FileSummary]:
        """Find files whose purpose or responsibilities match a concept.

        Simple substring matching against purpose + responsibilities.
        Designed for agent reasoning, not semantic search.
        """
        concept_lower = concept.lower()
        matches: list[FileSummary] = []
        for s in self._summaries.values():
            text = (s.purpose + " " + " ".join(s.responsibilities)).lower()
            if concept_lower in text:
                matches.append(s)
        return matches

    def files_by_category(self, category: str) -> list[FileSummary]:
        """Get all files in a given category."""
        return [s for s in self._summaries.values() if s.category == category]

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------

    def save_cache(self) -> None:
        """Persist summaries to .corecoder/file_summaries.json."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            fp: {
                "path": s.path,
                "purpose": s.purpose,
                "responsibilities": s.responsibilities,
                "key_symbols": s.key_symbols,
                "category": s.category,
                "file_type": s.file_type,
            }
            for fp, s in self._summaries.items()
        }
        self._cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_cache(self) -> bool:
        """Load cached summaries. Returns True if cache was loaded."""
        if not self._cache_path.exists():
            return False
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            for fp, d in data.items():
                self._summaries[fp] = FileSummary(
                    path=d["path"],
                    purpose=d.get("purpose", ""),
                    responsibilities=d.get("responsibilities", []),
                    key_symbols=d.get("key_symbols", []),
                    category=d.get("category", ""),
                    file_type=d.get("file_type", ""),
                )
            return len(self._summaries) > 0
        except (json.JSONDecodeError, KeyError):
            return False

    # ------------------------------------------------------------------
    # heuristic analysis
    # ------------------------------------------------------------------

    def _categorize(self, filepath: str, symbols: list[str]) -> str:
        """Categorize a file by path + symbol patterns."""
        fname = filepath.split("/")[-1].lower()

        # Path-based hints
        if "test" in fname or "test_" in fname or fname.startswith("test_"):
            return "test"
        if fname in ("main.py", "cli.py", "app.py", "run.py", "__main__.py"):
            return "cli"
        if fname in ("setup.py", "pyproject.toml", "Makefile", "Dockerfile"):
            return "config"
        if fname.endswith((".yaml", ".yml", ".toml", ".json", ".cfg", ".ini")):
            return "config"
        if "config" in fname or "settings" in fname or "constants" in fname:
            return "config"
        if fname in ("__init__.py",):
            return "package"

        # Symbol-based hints
        sym_set = set(s.lower() for s in symbols)
        if any(s in sym_set for s in ("main", "cli", "parse_args", "argparse")):
            return "cli"
        if any("router" in s or "route" in s or "endpoint" in s for s in sym_set):
            return "web"
        if any("model" in s or "schema" in s or "entity" in s for s in sym_set):
            return "data"
        if any("db" in s or "database" in s or "repository" in s for s in sym_set):
            return "data"
        if any("util" in s or "helper" in s for s in sym_set):
            return "utility"
        if any("service" in s or "handler" in s or "manager" in s for s in sym_set):
            return "core_logic"

        return "core_logic"

    def _infer_purpose(
        self, filepath: str, symbols: list[str], category: str
    ) -> str:
        """Infer a one-line purpose from file structure."""
        fname = filepath.split("/")[-1].replace(".py", "")

        if category == "cli":
            return f"CLI entry point for {self._guess_domain(filepath)}"
        if category == "test":
            domain = fname.replace("test_", "").replace("_test", "")
            return f"Tests for {domain}"
        if category == "config":
            return f"Configuration: {fname}"
        if category == "package":
            return f"Package init: {fname}"
        if category == "utility":
            return f"Utility functions: {', '.join(symbols[:3])}"
        if category == "data":
            return f"Data models/schemas: {', '.join(symbols[:3])}"
        if category == "web":
            return f"Web handlers/routes: {', '.join(symbols[:3])}"

        # core_logic — describe by key symbols
        exported = [s for s in symbols if not s.startswith("_")][:3]
        return f"Core module: {', '.join(exported)}" if exported else f"Module: {fname}"

    def _infer_responsibilities(
        self, filepath: str, symbols: list[str], category: str
    ) -> list[str]:
        """Infer a short list of responsibilities."""
        responsibilities: list[str] = []

        if category == "cli":
            responsibilities.append("Parse command-line arguments")
            responsibilities.append("Dispatch commands to handlers")
            if "main" in [s.lower() for s in symbols]:
                responsibilities.append("Application entry point")
        elif category == "test":
            responsibilities.append("Define test cases")
            if any("fixture" in s.lower() for s in symbols):
                responsibilities.append("Provide test fixtures")
        elif category == "data":
            responsibilities.append("Define data structures")
            if any("validate" in s.lower() for s in symbols):
                responsibilities.append("Validate data")
        elif category == "core_logic":
            # Extract from symbol names
            action_words = {
                "calc", "compute", "parse", "format", "convert", "validate",
                "execute", "run", "handle", "process", "build", "create",
                "update", "delete", "get", "set", "load", "save", "read", "write",
                "transform", "filter", "sort", "search", "find", "match",
            }
            for sym in symbols:
                sym_lower = sym.lower()
                for action in action_words:
                    if action in sym_lower:
                        responsibilities.append(f"{action} {self._guess_domain(filepath)}")
                        break
                if len(responsibilities) >= 3:
                    break

        if not responsibilities:
            responsibilities.append("Core business logic")

        return responsibilities[:5]

    def _key_symbols(self, symbols: list[str], category: str) -> list[str]:
        """Select the most important symbols for a file summary."""
        # Prioritize: public > dunder > private
        public = [s for s in symbols if not s.startswith("_")]
        if public:
            return public[:5]
        return symbols[:5]

    def _guess_domain(self, filepath: str) -> str:
        """Guess the domain name from the file path."""
        # e.g. "src/myapp/cli.py" → "myapp"
        parts = filepath.replace("\\", "/").split("/")
        for part in reversed(parts[:-1]):
            if part not in ("src", "lib", "app", "core", "."):
                return part
        return parts[-1].replace(".py", "") if parts else "application"

    @staticmethod
    def _extract_names(symbols: Any) -> list[str]:
        """Extract symbol names from the symbols JSON entry."""
        if isinstance(symbols, dict):
            return list(symbols.keys())
        if isinstance(symbols, list):
            return [
                s.get("name", "?")
                if isinstance(s, dict)
                else str(s)
                for s in symbols
            ]
        return []

    @staticmethod
    def _file_type(filepath: str) -> str:
        ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
        type_map = {
            "py": "python", "yaml": "yaml", "yml": "yaml",
            "toml": "toml", "json": "json", "md": "markdown",
            "cfg": "config", "ini": "config", "txt": "text",
        }
        return type_map.get(ext, ext)

    @staticmethod
    def _should_skip(filepath: str) -> bool:
        parts = filepath.replace("\\", "/").split("/")
        for part in parts:
            if part in ("__pycache__", ".corecoder", ".git", ".venv", "venv", "node_modules"):
                return True
        return filepath.endswith((".pyc", ".pyo", ".so", ".dll", ".pyd", ".exe"))
