"""Repository layer structured codebase memory.

Instead of dumping raw messages into the LLM context (the "chatbot toy"
approach), this module builds and maintains a structured index of the
project the agent is working on:

  repository_summary.md  - framework, ORM, entry points, conventions
  symbols.json           - classes, functions, their file locations
  dependencies.json      - package deps + internal import graph

The index lives in ``<project>/.corecoder/`` and is rebuilt incrementally
(via file mtime checks). The system prompt includes a condensed summary,
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

logger = logging.getLogger("corecoder.codebase")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
# Directories and file extensions to skip when scanning repos
_SKIP_DIRS = frozenset({
    "__pycache__", ".corecoder", ".git", ".venv", "venv", "node_modules",
    ".tox", "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
_SKIP_EXTENSIONS = (".pyc", ".pyo", ".so", ".dll", ".pyd", ".exe")

def should_skip_path(filepath: str) -> bool:
    """Check if a file path should be excluded from repo scanning.
    Shared utility — used by RepoIndex, SymbolOwnershipGraph,
    FileSummaryManager, and RepositoryContextRetriever.
    """
    parts = filepath.replace("\\", "/").split("/")
    for part in parts:
        if part in _SKIP_DIRS:
            return True
    return filepath.endswith(_SKIP_EXTENSIONS)

def _find_source_files(root: Path) -> list[Path]:
    """Find all source files, respecting common ignore patterns."""
    files: list[Path] = []
    for item in root.rglob("*"):
        if any(p in _SKIP_DIRS for p in item.parts):
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
def _safe_parse_python(filepath: Path) -> tuple[str, ast.AST | None]:
    try:
        source = filepath.read_text(encoding="utf-8-sig", errors="replace")
        return source, ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return "", None


def _node_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _node_name(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    if isinstance(expr, ast.Call):
        return _node_name(expr.func)
    if isinstance(expr, ast.Subscript):
        return _node_name(expr.value)
    return ""


def _format_signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    return_prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{return_prefix} {node.name}({', '.join(args)})"


def _extract_python_symbols(filepath: Path) -> dict[str, dict[str, Any]]:
    """Parse a Python file and return rich symbol metadata."""
    symbols: dict[str, dict[str, Any]] = {}
    source, tree = _safe_parse_python(filepath)
    if tree is None:
        return symbols

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": child.name,
                            "kind": "method",
                            "line": getattr(child, "lineno", 0),
                            "signature": _format_signature(child),
                        }
                    )
            doc = ast.get_docstring(node) or ""
            symbols[node.name] = {
                "kind": "class",
                "line": getattr(node, "lineno", 0),
                "signature": f"class {node.name}",
                "doc": doc,
                "methods": methods,
                "bases": [name for name in (_node_name(base) for base in node.bases) if name],
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            symbols[node.name] = {
                "kind": "function",
                "line": getattr(node, "lineno", 0),
                "signature": _format_signature(node),
                "doc": doc,
                "methods": [],
                "bases": [],
            }
    return symbols


def _extract_python_relations(filepath: Path) -> dict[str, list[str]]:
    """Extract imports, symbol references, calls, and inheritance hints."""
    source, tree = _safe_parse_python(filepath)
    if tree is None:
        return {
            "imports": [],
            "symbol_calls": [],
            "symbol_references": [],
            "inheritance": [],
        }

    imports: list[str] = []
    symbol_calls: list[str] = []
    symbol_references: list[str] = []
    inheritance: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        elif isinstance(node, ast.Call):
            name = _node_name(node.func)
            if name:
                symbol_calls.append(name)
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = _node_name(base)
                if name:
                    inheritance.append(name)
        elif isinstance(node, ast.Name):
            symbol_references.append(node.id)
        elif isinstance(node, ast.Attribute):
            name = _node_name(node)
            if name:
                symbol_references.append(name)

    def _dedup(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    return {
        "imports": _dedup(imports),
        "symbol_calls": _dedup(symbol_calls),
        "symbol_references": _dedup(symbol_references),
        "inheritance": _dedup(inheritance),
    }


def _module_name_for_path(root: Path, filepath: Path) -> str:
    rel = filepath.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_internal_imports(
    root: Path,
    py_files: list[Path],
    raw_imports: dict[str, list[str]],
) -> dict[str, list[str]]:
    module_to_file: dict[str, str] = {}
    for path in py_files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        module = _module_name_for_path(root, path)
        if module:
            module_to_file[module] = rel

    resolved: dict[str, list[str]] = {}
    for filepath, imports in raw_imports.items():
        matches: list[str] = []
        for imported in imports:
            candidates = [imported]
            if "." in imported:
                parts = imported.split(".")
                candidates.extend(".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1))
            for candidate in candidates:
                target = module_to_file.get(candidate)
                if target and target != filepath and target not in matches:
                    matches.append(target)
                    break
        resolved[filepath] = matches
    return resolved

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
            in_deps = False
            for line in text.splitlines():
                if line.strip().startswith("dependencies"):
                    in_deps = True
                    continue
                if in_deps:
                    if line.strip().startswith("[") or line.strip().startswith("#"):
                        break
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
        self._symbols: dict[str, dict[str, Any]] = {}
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
        raw_imports: dict[str, list[str]] = {}
        symbol_calls: dict[str, list[str]] = {}
        symbol_references: dict[str, list[str]] = {}
        inheritance: dict[str, list[str]] = {}
        for f in py_files:
            rel = str(f.relative_to(self.root))
            syms = _extract_python_symbols(f)
            if syms:
                self._symbols[rel] = syms
            relations = _extract_python_relations(f)
            imps = relations["imports"]
            raw_imports[rel] = imps
            all_imports.extend(imps)
            if relations["symbol_calls"]:
                symbol_calls[rel] = relations["symbol_calls"]
            if relations["symbol_references"]:
                symbol_references[rel] = relations["symbol_references"]
            if relations["inheritance"]:
                inheritance[rel] = relations["inheritance"]
        resolved_imports = _resolve_internal_imports(self.root, py_files, raw_imports)
        # dependencies
        declared = _extract_dependencies(self.root)
        self._deps = {
            "declared": declared,
            "imports": raw_imports,
            "internal_imports": resolved_imports,
            "resolved_internal_imports": resolved_imports,
            "symbol_calls": symbol_calls,
            "symbol_references": symbol_references,
            "inheritance": inheritance,
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
        lines = self._summary.strip().split("\n")
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
                info = syms[name]
                if isinstance(info, dict):
                    kind = str(info.get("kind", "symbol"))
                    line = int(info.get("line", 0) or 0)
                    methods = info.get("methods", [])
                    if kind == "class":
                        method_names = [
                            m if isinstance(m, str) else m.get("name", "")
                            for m in methods
                        ]
                        method_names = [m for m in method_names if m]
                        suffix = f" ({', '.join(method_names)})" if method_names else ""
                        results.append(f"{filepath}:{line}: class {name}{suffix}")
                    else:
                        results.append(f"{filepath}:{line}: {kind} {name}")
                else:
                    methods = info if isinstance(info, list) else []
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
                hits.append(f"{filepath} — {', '.join(i for i in imps if module in i)}")
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
            f"{self.root.name}\n",
            "",
            f"Path: `{self.root}`\n",
            f"Framework: {framework}\n",
            f"Entry points: {', '.join(entries) if entries else 'unknown'}\n",
            "",
            "Dependencies",
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
        shown = 0
        for filepath, syms in self._symbols.items():
            if shown >= 50:
                break
            for name in sorted(syms):
                if shown >= 50:
                    break
                info = syms[name]
                if isinstance(info, dict):
                    kind = str(info.get("kind", "symbol"))
                    methods = info.get("methods", [])
                    if kind == "class":
                        lines.append(f"- `{name}` ({len(methods)} methods) - `{filepath}`")
                    else:
                        lines.append(f"- `{name}()` - `{filepath}`")
                else:
                    methods = info if isinstance(info, list) else []
                    if methods:
                        lines.append(f"- `{name}` ({len(methods)} methods) - `{filepath}`")
                    else:
                        lines.append(f"- `{name}()` - `{filepath}`")
                shown += 1
        lines.extend([
            "",
            f"*{sum(len(v) for v in self._symbols.values())} symbols in ",
            f"{len(self._symbols)} files indexed.*",
        ])
        return "\n".join(lines)
