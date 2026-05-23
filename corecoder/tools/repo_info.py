"""Repository info tool — query the structured codebase index.

Lets the agent ask: "where is class X defined?", "what imports module Y?",
"what are the declared dependencies?", without needing to grep blindly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool

if TYPE_CHECKING:
    from ..repo_index import RepoIndex


class RepoInfoTool(Tool):
    name = "repo_info"
    description = (
        "Query the repository's structured index. Use this to find: "
        "where a symbol (class/function) is defined, which files import "
        "a module, what dependencies the project has, or read the "
        "full repository summary. Much faster and more accurate than grep."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["symbol", "imports", "dependencies", "summary"],
                "description": (
                    "What kind of query: 'symbol' to find a class/function, "
                    "'imports' to find files importing a module, "
                    "'dependencies' to list all declared deps, "
                    "'summary' to read the full repo summary"
                ),
            },
            "name": {
                "type": "string",
                "description": "Symbol name or module name (for symbol/imports query types)",
            },
        },
        "required": ["query_type"],
    }

    # set by Agent after construction
    _repo_index: RepoIndex | None = None

    def execute(self, query_type: str, name: str = "") -> str:
        idx = self._repo_index
        if idx is None:
            return "Error: repo index not available"

        if query_type == "symbol":
            if not name:
                return "Error: 'name' parameter required for symbol query"
            return idx.find_symbol(name)
        elif query_type == "imports":
            if not name:
                return "Error: 'name' parameter required for imports query"
            return idx.find_imports(name)
        elif query_type == "dependencies":
            return idx.list_dependencies()
        elif query_type == "summary":
            return idx.summary_full
        else:
            return f"Unknown query_type: {query_type}"
