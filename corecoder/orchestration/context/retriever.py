"""Repository-aware context retrieval.

Goes beyond naive grep by leveraging the existing repository index:
- Symbol proximity in the dependency graph
- Caller/callee relationships
- File-level dependency neighborhoods
- Architectural distance metrics

The retriever integrates with CoreCoder's RepoIndex for structured
symbol and dependency lookups, falling back to file glob when the
index is unavailable.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from corecoder.orchestration.context.models import (
    ContextFragment,
    ContextSource,
    ContextType,
    ContextRequest,
    ExecutionState,
)


@dataclass
class RetrievalOptions:
    """Controls for the retrieval process."""

    max_files: int = 10
    max_symbols: int = 20
    dependency_radius: int = 2    # How many hops in the dependency graph
    include_callers: bool = True
    include_callees: bool = True
    prefer_recently_modified: bool = True


class RepositoryContextRetriever:
    """Graph-aware repository context retrieval.

    Uses the structured repo index (symbols.json, dependencies.json)
    to find relevant files by dependency proximity rather than raw text
    matching.

    Usage:
        retriever = RepositoryContextRetriever(working_dir="/path/to/repo")
        fragments = retriever.retrieve(request)
    """

    def __init__(self, working_dir: str = "."):
        self._working_dir = Path(working_dir)
        self._index_dir = self._working_dir / ".corecoder"

        # Lazy-loaded index data
        self._symbols: dict[str, Any] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._summary: str = ""
        self._loaded = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        request: ContextRequest,
        options: RetrievalOptions | None = None,
    ) -> list[ContextFragment]:
        """Retrieve repository context — metadata only, no file contents.

        File CONTENTS belong in tool results (read_file), not in the
        pre-assembled prompt.  Dumping entire files into context wastes
        tokens on content the agent may already know from prior turns
        or may not need.

        Instead, we provide:
        - A structured file overview (names, symbols per file)
        - Dependency relationships
        - The agent decides which files to read via the read_file tool.
        """
        opts = options or RetrievalOptions()
        self._ensure_loaded()

        fragments: list[ContextFragment] = []

        # 1. Structured file overview — relevant files only, not the whole repo.
        # Match task keywords against file names to find relevant files.
        import re
        task_text = (request.task_title + " " + request.task_description + " " + request.goal).lower()
        keywords = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', task_text))

        if self._symbols:
            # Score files by keyword match
            scored_files: list[tuple[str, list[str], int]] = []
            for filepath, syms in self._symbols.items():
                if self._should_skip_file(filepath):
                    continue
                names = list(syms.keys()) if isinstance(syms, dict) else []
                if isinstance(syms, list):
                    names = [s.get("name", "?") if isinstance(s, dict) else str(s) for s in syms]
                stem = filepath.replace("\\", "/").split("/")[-1].replace(".py", "").lower()
                score = sum(1 for kw in keywords if kw in stem or kw in " ".join(names).lower())
                scored_files.append((filepath, names, score))

            # Keep top N by relevance score, then top N with no score
            scored_files.sort(key=lambda x: x[2], reverse=True)
            relevant = [(f, n) for f, n, s in scored_files if s > 0][:10]
            if len(relevant) < 5:
                rest = [(f, n) for f, n, s in scored_files if s == 0][:5]
                relevant.extend(rest)

            if relevant:
                lines = ["## Project Files (use read_file to see contents)"]
                for filepath, names in relevant[:12]:
                    if names:
                        lines.append(f"- {filepath} ({', '.join(names[:5])})")
                    else:
                        lines.append(f"- {filepath}")
                fragments.append(ContextFragment(
                    source=ContextSource.REPOSITORY,
                    type=ContextType.METADATA,
                    content="\n".join(lines),
                    priority=9,
                    relevance_score=0.95,
                    token_count=len("\n".join(lines)) // 3,
                ))

        # 2. Dependency relationships — what imports what
        if self._dependencies:
            internal = self._dependencies.get("internal_imports", {})
            if internal:
                dep_lines = ["## Key Imports"]
                for f, imps in list(internal.items())[:10]:
                    if self._should_skip_file(f):
                        continue
                    dep_lines.append(f"- {f} imports: {', '.join(imps[:5])}")
                if len(dep_lines) > 1:
                    fragments.append(ContextFragment(
                        source=ContextSource.DEPENDENCY_GRAPH,
                        type=ContextType.DEPENDENCY,
                        content="\n".join(dep_lines),
                        priority=7,
                        relevance_score=0.85,
                        token_count=len("\n".join(dep_lines)) // 3,
                    ))

        return fragments

    def _discover_files(self, request: ContextRequest, options: RetrievalOptions) -> list[str]:
        """Auto-discover relevant files when no focus_files are specified.

        Matches task title/description keywords against the file listing
        and symbol index.  Falls back to all known Python files if no
        keywords match.
        """
        files: list[str] = []
        title = (request.task_title + " " + request.task_description).lower()
        goal = request.goal.lower()

        # Extract keywords from task: split on spaces, punctuation, camelCase
        import re
        keywords = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', title + " " + goal))

        # Match keywords against known modules
        all_modules = list(self._symbols.keys()) if self._symbols else []
        for mod in all_modules:
            mod_stem = mod.replace("\\", "/").split("/")[-1].replace(".py", "").lower()
            for kw in keywords:
                if kw in mod_stem or mod_stem in kw:
                    files.append(mod)
                    break

        # If no matches, include all known Python files (max 3)
        if not files and all_modules:
            files = all_modules[:3]

        return files[:options.max_files]

    # ------------------------------------------------------------------
    # retrieval helpers
    # ------------------------------------------------------------------

    def _should_skip_file(self, file_path: str) -> bool:
        """Check if a file should be excluded from context."""
        parts = file_path.replace("\\", "/").split("/")
        # Skip binary artifacts and internal index files
        for part in parts:
            if part in ("__pycache__", ".corecoder", ".git", ".venv", "venv", "node_modules"):
                return True
        # Skip binary/compiled files
        if file_path.endswith((".pyc", ".pyo", ".so", ".dll", ".pyd", ".exe")):
            return True
        return False

    def _read_file_fragment(
        self, file_path: str, priority: int = 7, max_lines: int = 200
    ) -> ContextFragment | None:
        """Read a file and create a code fragment."""
        if self._should_skip_file(file_path):
            return None
        full_path = self._working_dir / file_path
        if not full_path.exists():
            return None
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        # Skip if content looks binary (null bytes or >30% non-printable)
        if "\x00" in content[:1000]:
            return None
        non_printable = sum(1 for c in content[:500] if ord(c) < 32 and c not in "\n\r\t")
        if non_printable > len(content[:500]) * 0.3:
            return None

        # Truncate if too large
        lines = content.split("\n")
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

        return ContextFragment(
            source=ContextSource.REPOSITORY,
            type=ContextType.CODE,
            content=f"## File: {file_path}\n```\n{content}\n```",
            priority=priority,
            relevance_score=0.7,
            token_count=len(content) // 3,
            origin_file=file_path,
            metadata={"file_path": file_path, "line_count": len(lines)},
        )

    def _retrieve_symbol_context(
        self, sym_name: str, options: RetrievalOptions
    ) -> list[ContextFragment]:
        """Retrieve context around a specific symbol."""
        fragments: list[ContextFragment] = []

        if not self._symbols:
            return fragments

        # Find matching symbols
        matches = []
        sym_lower = sym_name.lower()
        for file_path, symbols in self._symbols.items():
            if isinstance(symbols, list):
                for s in symbols:
                    if isinstance(s, dict) and sym_lower in s.get("name", "").lower():
                        matches.append((file_path, s))
            elif isinstance(symbols, dict):
                if sym_lower in symbols.get("name", "").lower():
                    matches.append((file_path, symbols))

        for file_path, sym_info in matches[:options.max_symbols]:
            # Read the file containing this symbol
            frag = self._read_file_fragment(file_path, priority=8)
            if frag:
                frag.type = ContextType.SYMBOL_DEF
                frag.metadata["symbol_name"] = sym_info.get("name", sym_name) if isinstance(sym_info, dict) else sym_name
                fragments.append(frag)

        return fragments

    def _expand_dependency_neighborhood(
        self, seed_files: set[str], radius: int
    ) -> list[str]:
        """Expand a set of files by following dependency edges."""
        if not self._dependencies or not seed_files:
            return list(seed_files)

        neighborhood = set(seed_files)
        frontier = set(seed_files)

        for _ in range(radius):
            next_frontier: set[str] = set()
            for f in frontier:
                # Normalize path separators
                f_normalized = f.replace("\\", "/")
                deps = self._dependencies.get(f_normalized, [])
                for dep in deps:
                    if dep not in neighborhood:
                        neighborhood.add(dep)
                        next_frontier.add(dep)
                # Also check: which files depend on f?
                for other, other_deps in self._dependencies.items():
                    if f_normalized in other_deps and other not in neighborhood:
                        neighborhood.add(other)
                        next_frontier.add(other)
            frontier = next_frontier
            if not frontier:
                break

        return list(neighborhood)

    def _find_related_symbols(
        self, focus_symbols: set[str], options: RetrievalOptions
    ) -> list[ContextFragment]:
        """Find symbols related to the focus set by shared files."""
        fragments: list[ContextFragment] = []
        related_files: set[str] = set()

        for file_path, symbols in self._symbols.items():
            if not isinstance(symbols, list):
                continue
            for s in symbols:
                if not isinstance(s, dict):
                    continue
                name = s.get("name", "").lower()
                if any(fs in name for fs in focus_symbols):
                    related_files.add(file_path)
                    break

        for fpath in list(related_files)[:options.max_files]:
            frag = self._read_file_fragment(fpath, priority=6)
            if frag:
                fragments.append(frag)

        return fragments

    # ------------------------------------------------------------------
    # index loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the repository index from .corecoder/."""
        if self._loaded:
            return
        self._loaded = True

        # Load symbols
        symbols_path = self._index_dir / "symbols.json"
        if symbols_path.exists():
            try:
                self._symbols = json.loads(symbols_path.read_text(encoding="utf-8"))
            except Exception:
                self._symbols = {}

        # Load dependencies
        deps_path = self._index_dir / "dependencies.json"
        if deps_path.exists():
            try:
                self._dependencies = json.loads(deps_path.read_text(encoding="utf-8"))
            except Exception:
                self._dependencies = {}

        # Load summary
        summary_path = self._index_dir / "repository_summary.md"
        if summary_path.exists():
            try:
                self._summary = summary_path.read_text(encoding="utf-8")
            except Exception:
                self._summary = ""

    def invalidate_cache(self) -> None:
        """Force reload of index data on next retrieval."""
        self._loaded = False
        self._symbols = {}
        self._dependencies = {}
        self._summary = ""
