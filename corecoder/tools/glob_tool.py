"""File pattern matching — auto-filters noise directories and binary artifacts."""

from pathlib import Path
from .base import Tool

# Directories and file extensions that are never useful to the agent.
# Filtered BEFORE sorting/truncation so the agent sees only project files.
_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", ".env", "__pycache__",
    ".corecoder", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox",
    "dist", "build", "egg-info", "*.egg-info",
})
_SKIP_EXTS = frozenset({
    ".pyc", ".pyo", ".so", ".dll", ".pyd", ".exe",
    ".obj", ".o", ".a", ".lib",
})


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py'). "
        "Auto-filters: .venv, .git, __pycache__, .corecoder, node_modules, "
        "binary artifacts (.pyc, .so, .dll, etc)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = [p for p in base.glob(pattern) if not _is_noise(p)]
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:
            return f"Error: {e}"


def _is_noise(p: Path) -> bool:
    """Return True if this path should be hidden from the agent."""
    # Check each path component against the skip-dirs set
    for part in p.parts:
        if part in _SKIP_DIRS:
            return True
        # Match patterns like "*.egg-info"
        if part.endswith(".egg-info"):
            return True
    # Check extension
    if p.suffix.lower() in _SKIP_EXTS:
        return True
    return False
