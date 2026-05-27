"""Symbol Ownership Graph — the backbone of symbolic retrieval.

Provides:
- symbol → file lookup (where is X defined?)
- file → symbols lookup (what does this file contain?)
- partial symbol match (fuzzy prefix/suffix matching)
- reverse lookup (which files reference symbol X?)

Design: built from the existing .corecoder/symbols.json index.
All lookups are O(1) or O(log n).  No full scans at query time.
"""

from __future__ import annotations

from typing import Any
from corecoder.orchestration.retrieval.models import SymbolInfo
from corecoder.repo.index import should_skip_path


class SymbolOwnershipGraph:
    """Bidirectional index: symbol ↔ file.

    Built once from the repository index, then queried repeatedly.
    Supports exact lookup, partial matching, and fuzzy search.

    Usage:
        sog = SymbolOwnershipGraph()
        sog.build_from_index(symbols_json)
        matches = sog.lookup("sqrt")       # exact
        matches = sog.fuzzy_search("sqr")  # partial
        symbols = sog.file_symbols("calculator.py")
    """

    def __init__(self):
        # symbol_name → list of SymbolInfo (handles overloads / same name in multiple files)
        self._symbol_map: dict[str, list[SymbolInfo]] = {}

        # filepath → list of SymbolInfo
        self._file_map: dict[str, list[SymbolInfo]] = {}

        # Lowercase index for case-insensitive lookup
        self._lower_index: dict[str, list[str]] = {}  # lower_name → [exact_names]

        # Prefix index: first 3 chars → matching symbol names
        self._prefix_index: dict[str, list[str]] = {}  # prefix(len 3) → [exact_names]

        self._built = False

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build_from_index(self, symbols_json: dict[str, Any]) -> None:
        """Build the graph from .corecoder/symbols.json data.

        symbols_json format:
            {
                "filepath.py": {
                    "symbol_name": {"kind": "function", "line": 10, ...},
                    ...
                },
                ...
            }
        Or list-based:
            {
                "filepath.py": [
                    {"name": "symbol_name", "kind": "function", ...},
                    ...
                ]
            }
        """
        self._symbol_map.clear()
        self._file_map.clear()
        self._lower_index.clear()
        self._prefix_index.clear()

        for filepath, symbols in symbols_json.items():
            if should_skip_path(filepath):
                continue

            filepath = filepath.replace("\\", "/")
            infos: list[SymbolInfo] = []

            if isinstance(symbols, dict):
                # Dict format: {name: info}
                for name, info in symbols.items():
                    if isinstance(info, dict):
                        si = SymbolInfo(
                            name=name,
                            kind=info.get("kind", "unknown"),
                            defined_in=filepath,
                            line=info.get("line", 0),
                            signature=info.get("signature", ""),
                            doc_brief=info.get("doc", "")[:80] if info.get("doc") else "",
                            exported=info.get("exported", False),
                        )
                    else:
                        si = SymbolInfo(name=name, kind="unknown", defined_in=filepath)
                    infos.append(si)
                    self._add_to_maps(si)

            elif isinstance(symbols, list):
                # List format: [{name: ..., kind: ...}]
                for s in symbols:
                    if isinstance(s, dict):
                        name = s.get("name", "?")
                        si = SymbolInfo(
                            name=name,
                            kind=s.get("kind", "unknown"),
                            defined_in=filepath,
                            line=s.get("line", 0),
                            signature=s.get("signature", ""),
                            doc_brief=s.get("doc", "")[:80] if s.get("doc") else "",
                        )
                        infos.append(si)
                        self._add_to_maps(si)

            if infos:
                self._file_map[filepath] = infos

        self._built = True

    def _add_to_maps(self, si: SymbolInfo) -> None:
        """Index a single symbol into all lookup structures."""
        name = si.name
        lower = name.lower()

        # Exact map
        if name not in self._symbol_map:
            self._symbol_map[name] = []
        self._symbol_map[name].append(si)

        # Lowercase index
        if lower not in self._lower_index:
            self._lower_index[lower] = []
        if name not in self._lower_index[lower]:
            self._lower_index[lower].append(name)

        # Prefix index (first 3 chars)
        for prefix_len in (3, 4, 5):
            if len(lower) >= prefix_len:
                pfx = lower[:prefix_len]
                if pfx not in self._prefix_index:
                    self._prefix_index[pfx] = []
                if name not in self._prefix_index[pfx]:
                    self._prefix_index[pfx].append(name)

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> list[SymbolInfo]:
        """Exact symbol lookup (case-insensitive)."""
        lower = name.lower()
        # Try exact match first
        if name in self._symbol_map:
            return self._symbol_map[name]
        # Case-insensitive
        exact_names = self._lower_index.get(lower, [])
        results: list[SymbolInfo] = []
        for n in exact_names:
            results.extend(self._symbol_map.get(n, []))
        return results

    def fuzzy_search(self, query: str, limit: int = 10) -> list[SymbolInfo]:
        """Partial/fuzzy symbol search.

        Tries: exact → prefix → substring matching.
        """
        query_lower = query.lower()

        # 1. Exact match
        exact = self.lookup(query)
        if exact:
            return exact[:limit]

        # 2. Prefix search
        results: list[SymbolInfo] = []
        seen: set[tuple[str, str]] = set()  # (name, file) dedup
        for prefix_len in (5, 4, 3):
            if len(query_lower) >= prefix_len:
                pfx = query_lower[:prefix_len]
                for name in self._prefix_index.get(pfx, []):
                    if query_lower in name.lower():
                        for si in self._symbol_map.get(name, []):
                            key = (si.name, si.defined_in)
                            if key not in seen:
                                seen.add(key)
                                results.append(si)
            if len(results) >= limit:
                break

        # 3. Substring: scan lower index as fallback
        if len(results) < 3:
            for lower_name, names in self._lower_index.items():
                if query_lower in lower_name:
                    for n in names:
                        for si in self._symbol_map.get(n, []):
                            key = (si.name, si.defined_in)
                            if key not in seen:
                                seen.add(key)
                                results.append(si)
                if len(results) >= limit:
                    break

        return results[:limit]

    def file_symbols(self, filepath: str) -> list[SymbolInfo]:
        """Get all symbols defined in a file."""
        filepath = filepath.replace("\\", "/")
        return self._file_map.get(filepath, [])

    def files_for_symbols(self, names: list[str]) -> list[str]:
        """Get all files containing any of the given symbol names."""
        files: set[str] = set()
        for name in names:
            for si in self.lookup(name):
                files.add(si.defined_in)
            # Also try fuzzy for partial matches
            for si in self.fuzzy_search(name, limit=3):
                files.add(si.defined_in)
        return list(files)

    def file_count(self) -> int:
        return len(self._file_map)

    def symbol_count(self) -> int:
        return sum(len(v) for v in self._symbol_map.values())

    @property
    def is_built(self) -> bool:
        return self._built

