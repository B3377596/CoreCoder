"""Bidirectional Dependency Graph ?*forward + reverse edges pre-computed.

Replaces the old flat dict-based dependency model with a proper
bidirectional structure.  Both directions are pre-computed at build
time, so neighborhood expansion is O(edges) instead of O(n?).

Paths are normalized to forward slashes.
"""

from __future__ import annotations

from corecoder.retrieval.models import BidirectionalDepGraph


def build_dependency_graph(
    dependencies_json: dict,
) -> BidirectionalDepGraph:
    """Build a bidirectional dependency graph from .corecoder/dependencies.json.

    Expected JSON format:
        {
            "internal_imports": {
                "file.py": ["other.py", "module.py"],
                ...
            }
        }

    Produces both:
    - imports: file ?*what it imports
    - imported_by: file ?*who imports it
    """
    raw_imports = (
        dependencies_json.get("resolved_internal_imports")
        or dependencies_json.get("internal_imports", {})
    )
    if not raw_imports:
        return BidirectionalDepGraph()

    # Normalize
    imports: dict[str, list[str]] = {}
    imported_by: dict[str, list[str]] = {}

    for filepath, deps in raw_imports.items():
        filepath = filepath.replace("\\", "/")
        imports[filepath] = [d.replace("\\", "/") for d in deps]

    # Build reverse index
    for filepath, deps in imports.items():
        for dep in deps:
            dep_normalized = dep.replace("\\", "/")
            if dep_normalized not in imported_by:
                imported_by[dep_normalized] = []
            if filepath not in imported_by[dep_normalized]:
                imported_by[dep_normalized].append(filepath)

    return BidirectionalDepGraph(imports=imports, imported_by=imported_by)
