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
        """Retrieve repository context fragments for a request."""
        opts = options or RetrievalOptions()
        self._ensure_loaded()

        fragments: list[ContextFragment] = []

        # 1. Repository summary (always include, but compressed for non-planning)
        if self._summary:
            if request.execution_state == ExecutionState.PLANNING:
                fragments.append(ContextFragment(
                    source=ContextSource.REPOSITORY,
                    type=ContextType.SUMMARY,
                    content=f"## Repository Overview\n{self._summary[:3000]}",
                    priority=10,
                    relevance_score=1.0,
                    token_count=len(self._summary) // 3,
                ))
            else:
                # In coding mode, just include a one-line summary
                first_line = self._summary.split("\n")[0] if self._summary else ""
                if first_line:
                    fragments.append(ContextFragment(
                        source=ContextSource.REPOSITORY,
                        type=ContextType.SUMMARY,
                        content=f"Project: {first_line[:200]}",
                        priority=5,
                        relevance_score=0.5,
                        token_count=len(first_line) // 3,
                    ))

        # 2. Focus files (from the request's focus_files)
        for file_path in request.focus_files[:opts.max_files]:
            frag = self._read_file_fragment(file_path)
            if frag:
                fragments.append(frag)

        # 3. Symbol lookup for focus_symbols
        for sym_name in request.focus_symbols[:opts.max_symbols]:
            sym_frags = self._retrieve_symbol_context(sym_name, opts)
            fragments.extend(sym_frags)

        # 4. Dependency neighborhood for files we already identified
        focus_file_set = set(request.focus_files)
        if opts.dependency_radius > 0 and self._dependencies:
            neighbor_files = self._expand_dependency_neighborhood(
                focus_file_set, radius=opts.dependency_radius
            )
            for fpath in neighbor_files[:opts.max_files]:
                if fpath not in focus_file_set:
                    frag = self._read_file_fragment(fpath, priority=5)
                    if frag:
                        fragments.append(frag)

        # 5. Symbol neighborhood — symbols near the focus symbols
        focus_symbol_set = set(s.lower() for s in request.focus_symbols)
        if focus_symbol_set and self._symbols:
            related = self._find_related_symbols(focus_symbol_set, opts)
            fragments.extend(related)

        return fragments

    # ------------------------------------------------------------------
    # retrieval helpers
    # ------------------------------------------------------------------

    def _read_file_fragment(
        self, file_path: str, priority: int = 7, max_lines: int = 200
    ) -> ContextFragment | None:
        """Read a file and create a code fragment."""
        full_path = self._working_dir / file_path
        if not full_path.exists():
            return None
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
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
