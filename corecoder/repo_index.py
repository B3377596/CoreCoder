"""Repository layer — structured codebase memory.

Instead of dumping raw messages into the LLM context (the "chatbot toy"
approach), this module builds and maintains a structured index of the
project the agent is working on:

  repository_summary.md  – framework, ORM, entry points, conventions
  symbols.json           – classes, functions, their file locations
  dependencies.json      – package deps + internal import graph

The index lives in ``<project>/.corecoder/`` and is rebuilt incrementally
(via file mtime checks).  The system prompt includes a condensed summary,
and a ``repo_info`` tool lets the agent query symbols and structure.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("corecoder.repo")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_source_files(root: Path) -> list[Path]:
    """Find all source files, respecting common ignore patterns."""
    ignore = {".git", ".venv", "venv", "node_modules", "__pycache__",
              ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
              ".ruff_cache", ".corecoder"}
    files: list[Path] = []
    for item in root.rglob("*"):
        if any(p in ignore for p in item.parts):
            continue
        if item.is_file():
            files.append(item)
    return files


def _file_hash(path: Path) -> str:
    """Quick hash based on mtime + size — not cryptographic, just for change detection."""
    try:
        stat = path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------


def _extract_python_symbols(filepath: Path) -> dict[str, list[str]]:
    """Parse a Python file and return {class_name: [methods], func_name: []}."""
    symbols: dict[str, list[str]] = {}
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, UnicodeDecodeError):
        return symbols

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            symbols[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # top-level function (not method)
            if node.name not in symbols:
                symbols[node.name] = []
    return symbols


def _extract_python_imports(filepath: Path) -> list[str]:
    """Extract import targets from a Python file."""
    imports: list[str] = []
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return imports


# ---------------------------------------------------------------------------
# dependency extraction
# ---------------------------------------------------------------------------


def _extract_dependencies(root: Path) -> list[str]:
    """Extract declared dependencies from common manifest files."""
    deps: list[str] = []

    # pyproject.toml — only match inside [project] dependencies
    toml = root / "pyproject.toml"
    if toml.exists():
        try:
            text = toml.read_text(encoding="utf-8")
            # find the [project] section and extract its dependencies block
            in_deps = False
            for line in text.splitlines():
                if line.strip().startswith("dependencies"):
                    in_deps = True
                    continue
                if in_deps:
                    if line.strip().startswith("[") or line.strip().startswith("#"):
                        break  # end of dependencies section
                    # match "pkg>=version" or "pkg"
                    m = re.match(r'\s*"([\w-]+)[^"]*"', line)
                    if m:
                        deps.append(m.group(1))
        except OSError:
            pass

    # requirements.txt
    req = root / "requirements.txt"
    if req.exists():
        try:
            for line in req.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    dep = re.split(r"[><=!~]", line)[0].strip()
                    deps.append(dep)
        except OSError:
            pass

    # package.json
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            for key in ("dependencies", "devDependencies"):
                deps.extend(data.get(key, {}).keys())
        except (json.JSONDecodeError, OSError):
            pass

    return sorted(set(deps))


# ---------------------------------------------------------------------------
# summary generation
# ---------------------------------------------------------------------------


def _detect_framework(root: Path, imports: list[str]) -> str:
    """Guess the framework from imports and config files."""
    all_imports = set(imports)
    signals: list[str] = []

    # Python web frameworks
    if "fastapi" in all_imports:
        signals.append("FastAPI")
    if "flask" in all_imports:
        signals.append("Flask")
    if "django" in all_imports:
        signals.append("Django")
    if "starlette" in all_imports:
        signals.append("Starlette")

    # Python ORMs
    if "sqlalchemy" in all_imports:
        signals.append("SQLAlchemy")
    if "tortoise" in all_imports:
        signals.append("Tortoise ORM")
    if "pony" in all_imports:
        signals.append("Pony ORM")

    # Test frameworks
    if "pytest" in all_imports:
        signals.append("pytest")

    # CLI / tools
    if "click" in all_imports:
        signals.append("Click")
    if "typer" in all_imports:
        signals.append("Typer")
    if "rich" in all_imports:
        signals.append("Rich")

    # Node
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "next" in all_deps:
                signals.append("Next.js")
            if "react" in all_deps:
                signals.append("React")
            if "express" in all_deps:
                signals.append("Express")
        except (json.JSONDecodeError, OSError):
            pass

    return ", ".join(signals) if signals else "unknown"


def _detect_entry_points(root: Path) -> list[str]:
    """Find likely entry points."""
    entries = []
    for name in ["main.py", "app.py", "cli.py", "run.py", "index.ts", "index.js",
                 "__main__.py", "server.py", "manage.py"]:
        if (root / name).exists():
            entries.append(name)
    # also check src/<name>/__main__.py patterns
    for pattern in ["__main__.py"]:
        for path in root.rglob(pattern):
            rel = str(path.relative_to(root))
            if any(p in ("node_modules", ".venv", ".git") for p in path.parts):
                continue
            entries.append(rel)
    return entries[:5]


# ---------------------------------------------------------------------------
# RepoIndex
# ---------------------------------------------------------------------------


class RepoIndex:
    """Structured knowledge about the repository the agent is working in.

    The index is stored under ``<project>/.corecoder/``:
      - repository_summary.md
      - symbols.json  ({file: {name: [methods]}})
      - dependencies.json  ({declared: [...], internal_imports: {file: [...]}})
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.index_dir = self.root / ".corecoder"
        self._summary: str = ""
        self._symbols: dict[str, dict[str, list[str]]] = {}
        self._deps: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # build / load
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Scan the repo and build all index files from scratch."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building repo index for %s", self.root)

        source_files = _find_source_files(self.root)
        py_files = [f for f in source_files if f.suffix == ".py"]

        # symbols
        self._symbols = {}
        all_imports: list[str] = []
        internal_imports: dict[str, list[str]] = {}

        for f in py_files:
            rel = str(f.relative_to(self.root))
            syms = _extract_python_symbols(f)
            if syms:
                self._symbols[rel] = syms
            imps = _extract_python_imports(f)
            if imps:
                internal_imports[rel] = imps
                all_imports.extend(imps)

        # dependencies
        declared = _extract_dependencies(self.root)
        self._deps = {
            "declared": declared,
            "internal_imports": internal_imports,
        }

        # summary
        framework = _detect_framework(self.root, all_imports)
        entries = _detect_entry_points(self.root)
        self._summary = self._render_summary(framework, declared, entries)

        # write to disk
        summary_path = self.index_dir / "repository_summary.md"
        summary_path.write_text(self._summary, encoding="utf-8")
        (self.index_dir / "symbols.json").write_text(
            json.dumps(self._symbols, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.index_dir / "dependencies.json").write_text(
            json.dumps(self._deps, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Repo index built: %d symbols in %d files, %d deps",
                     sum(len(v) for v in self._symbols.values()),
                     len(self._symbols), len(declared))

    def load(self) -> bool:
        """Load existing index from disk. Returns True if index exists and loads."""
        summary_path = self.index_dir / "repository_summary.md"
        symbols_path = self.index_dir / "symbols.json"
        deps_path = self.index_dir / "dependencies.json"

        if not summary_path.exists():
            return False

        try:
            self._summary = summary_path.read_text(encoding="utf-8")
            self._symbols = json.loads(symbols_path.read_text(encoding="utf-8"))
            self._deps = json.loads(deps_path.read_text(encoding="utf-8"))
            return True
        except (OSError, json.JSONDecodeError):
            return False

    def needs_rebuild(self) -> bool:
        """Check if any source file has changed since last index build."""
        stamp_file = self.index_dir / ".index_stamp"
        if not stamp_file.exists():
            return True
        try:
            stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True

        source_files = _find_source_files(self.root)
        for f in source_files:
            rel = str(f.relative_to(self.root))
            if stamp.get(rel) != _file_hash(f):
                return True
        return False

    def save_stamp(self) -> None:
        """Record current file hashes so needs_rebuild() works next time."""
        source_files = _find_source_files(self.root)
        stamp = {str(f.relative_to(self.root)): _file_hash(f) for f in source_files}
        (self.index_dir / ".index_stamp").write_text(
            json.dumps(stamp, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def summary(self) -> str:
        """Short summary for injection into the system prompt."""
        if not self._summary:
            return ""
        # return first section as a condensed hint
        lines = self._summary.strip().split("\n")
        # skip title line
        body = [l for l in lines if not l.startswith("# ") and not l.startswith(">")]
        return "\n".join(body[:30])

    @property
    def summary_full(self) -> str:
        return self._summary

    def find_symbol(self, name: str) -> str:
        """Find where a symbol is defined. Returns file:line hint."""
        results: list[str] = []
        for filepath, syms in self._symbols.items():
            if name in syms:
                methods = syms[name]
                if methods:
                    results.append(f"{filepath}: class {name} ({', '.join(methods)})")
                else:
                    results.append(f"{filepath}: def {name}()")
                if len(results) >= 10:
                    break
        if not results:
            return f"Symbol '{name}' not found in index."
        return "\n".join(results)

    def find_imports(self, module: str) -> str:
        """Find all files that import a given module."""
        hits = []
        for filepath, imps in self._deps.get("internal_imports", {}).items():
            if any(module in imp for imp in imps):
                hits.append(f"{filepath} ← {', '.join(i for i in imps if module in i)}")
                if len(hits) >= 20:
                    break
        return "\n".join(hits) if hits else f"No imports of '{module}' found."

    def list_dependencies(self) -> str:
        """List all declared dependencies."""
        deps = self._deps.get("declared", [])
        return "\n".join(f"- {d}" for d in deps) if deps else "(none)"

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _render_summary(self, framework: str, deps: list[str],
                        entries: list[str]) -> str:
        """Build the repository_summary.md content."""
        lines = [
            f"# {self.root.name}",
            "",
            f"**Path:** `{self.root}`",
            f"**Framework:** {framework}",
            f"**Entry points:** {', '.join(entries) if entries else 'unknown'}",
            "",
            "## Dependencies",
            "",
        ]
        for d in deps[:30]:
            lines.append(f"- {d}")
        if len(deps) > 30:
            lines.append(f"- ... and {len(deps) - 30} more")
        lines.extend([
            "",
            "## Key Symbols",
            "",
        ])
        # top-level classes and functions only
        shown = 0
        for filepath, syms in self._symbols.items():
            if shown >= 50:
                break
            for name in sorted(syms):
                if shown >= 50:
                    break
                methods = syms[name]
                if methods:
                    lines.append(f"- `{name}` ({len(methods)} methods) — `{filepath}`")
                else:
                    lines.append(f"- `{name}()` — `{filepath}`")
                shown += 1
        lines.extend([
            "",
            f"*{sum(len(v) for v in self._symbols.values())} symbols in "
            f"{len(self._symbols)} files indexed.*",
        ])
        return "\n".join(lines)
