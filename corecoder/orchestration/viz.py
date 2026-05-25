"""Terminal visualization for task graphs.

Renders a DAG as a text tree with status indicators, suitable for
display in a terminal REPL.  Uses ASCII-safe characters by default
with Unicode fallback when the terminal supports it.

Example output:
    Plan: Build a CLI calculator (4 tasks)
    ------------------------------------------------------------
      [OK] Create project structure
      |
      [>>] Implement core logic
      |--[  ] Add CLI interface
      `--[  ] Write tests
"""

from __future__ import annotations

import os
import sys

from corecoder.orchestration.dag.models import TaskNode, TaskStatus
from corecoder.orchestration.dag.graph import TaskGraph


def _use_unicode() -> bool:
    """Detect whether the terminal supports Unicode output."""
    # Windows legacy terminal (cmd.exe, PS without UTF-8) can't handle Unicode
    if sys.platform == "win32":
        codepage = os.environ.get("PYTHONIOENCODING", "")
        if "utf" not in codepage.lower():
            # Check if we're in Windows Terminal (supports Unicode)
            if os.environ.get("WT_SESSION"):
                return True
            return False
    return True


_UNICODE = _use_unicode()

# Status → icon mapping for terminal display
if _UNICODE:
    STATUS_ICONS: dict[str, str] = {
        "pending": "○",
        "ready": "◉",
        "running": "▶",
        "success": "✓",
        "failed": "✗",
        "blocked": "⊘",
        "skipped": "⊝",
    }
else:
    STATUS_ICONS: dict[str, str] = {
        "pending": "[  ]",
        "ready": "[..]",
        "running": "[>>]",
        "success": "[OK]",
        "failed": "[XX]",
        "blocked": "[BL]",
        "skipped": "[SK]",
    }

# Status → color name (for rich console rendering)
STATUS_COLORS: dict[str, str] = {
    "pending": "dim",
    "ready": "cyan",
    "running": "bold yellow",
    "success": "green",
    "failed": "red",
    "blocked": "dim",
    "skipped": "dim yellow",
}

# Tree-drawing characters — use Unicode when available, ASCII otherwise
if _UNICODE:
    TEE = "├──"
    ELBOW = "└──"
    PIPE = "│  "
    SPACE = "   "
    ARROW = "──▶"
    HLINE = "━"
else:
    TEE = "|--"
    ELBOW = "`--"
    PIPE = "|  "
    SPACE = "   "
    ARROW = "-->"
    HLINE = "-"


def status_icon(node: TaskNode) -> str:
    """Return the display icon for a node's current status."""
    status = node.status.value if isinstance(node.status, TaskStatus) else str(node.status)
    return STATUS_ICONS.get(status, "?")


def status_color(node: TaskNode) -> str:
    """Return the rich color for a node's current status."""
    status = node.status.value if isinstance(node.status, TaskStatus) else str(node.status)
    return STATUS_COLORS.get(status, "")


def render_graph(graph: TaskGraph, goal: str = "", max_width: int = 80) -> str:
    """Render a TaskGraph as a terminal tree.

    Uses a simple layout algorithm:
    1. Topological sort determines vertical ordering
    2. Each node is indented based on its depth from roots
    3. Edges are drawn as vertical pipes and horizontal connectors

    Returns a string suitable for printing to a terminal.
    """
    if graph.node_count == 0:
        return "(empty graph)"

    lines: list[str] = []

    # Header
    header = f"Plan: {goal}" if goal else f"Task Graph: {graph.name}"
    success, failed, running, pending = graph.progress()
    done = success + failed
    total = graph.node_count
    lines.append(header + f"  [{done}/{total}]")
    lines.append(HLINE * min(max_width, 60))

    # Build depth map: how many layers of dependencies above each node
    depths = _compute_depths(graph)

    # Group nodes by depth for layout
    nodes_by_depth: dict[int, list[TaskNode]] = {}
    for node in graph.nodes.values():
        d = depths.get(node.id, 0)
        nodes_by_depth.setdefault(d, []).append(node)

    max_depth = max(nodes_by_depth) if nodes_by_depth else 0

    # Build a tree-like representation
    # For each node, we need to know: who are its children (dependents)
    # and what tree prefix to use

    # Calculate positions (simple layered layout)
    # Each node gets a line, edges are drawn between layers

    # Use topological order, but group by depth
    ordered = graph.topological_sort()

    # For each node, track which column (position within its depth) it occupies
    positions: dict[str, int] = {}
    for d in range(max_depth + 1):
        for i, node in enumerate(nodes_by_depth.get(d, [])):
            positions[node.id] = i

    # Track which parent pipes need to continue through each depth
    # For each depth, track which column positions have active parents
    active_parents_by_depth: dict[int, set[int]] = {d: set() for d in range(max_depth + 2)}

    # Pre-compute: for each node, which children (dependents) does it have
    children_map: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        children_map[node.id] = graph.get_dependents(node.id)

    # Pre-compute parent positions for each node
    parent_depths: dict[str, set[int]] = {}
    for node in graph.nodes.values():
        deps = graph.get_dependencies(node.id)
        parent_depths[node.id] = {depths.get(d, 0) for d in deps}

    # For each depth level, track which parent column positions are active
    for node in ordered:
        d = depths.get(node.id, 0)
        for child_id in children_map.get(node.id, []):
            child_depth = depths.get(child_id, d + 1)
            for level in range(d + 1, child_depth + 1):
                active_parents_by_depth[level].add(positions[node.id])

    # Now render each node
    rendered: set[str] = set()
    for node in ordered:
        d = depths.get(node.id, 0)
        pos = positions.get(node.id, 0)

        # Build prefix: for each depth level up to current, show pipes or spaces
        prefix_parts: list[str] = []
        if d > 0:
            for level in range(d):
                # Check if any parent at this level has us as a descendant
                active = active_parents_by_depth.get(level + 1, set())
                # We need to show pipe if this column or nearby columns are active
                # Simplified: show pipe if any parent pipe passes through
                if active:
                    prefix_parts.append(PIPE)
                else:
                    prefix_parts.append(SPACE)

        # Determine connector
        siblings = nodes_by_depth.get(d, [])
        my_index = next((i for i, n in enumerate(siblings) if n.id == node.id), -1)
        is_last = my_index == len(siblings) - 1
        connector = ELBOW if is_last else TEE

        prefix = "".join(prefix_parts)
        if d > 0:
            line = f"{prefix}{connector} "
        else:
            line = "  "

        icon = status_icon(node)
        line += f"{icon} {node.title}"

        # Add duration if completed
        if node.result and node.result.duration_ms > 0:
            line += f"  [{_fmt_duration(node.result.duration_ms)}]"

        if node.retry_count > 0:
            line += f"  (retry {node.retry_count})"

        lines.append(line)
        rendered.add(node.id)

        # Track active parent pipes for our children
        for child_id in children_map.get(node.id, []):
            child_depth = depths.get(child_id, d + 1)
            for level in range(d + 1, child_depth + 1):
                active_parents_by_depth.setdefault(level, set()).add(pos)

    # Summary line
    success, failed, _, _ = graph.progress()
    summary_parts = []
    if success:
        summary_parts.append(f"[green]✓ {success} succeeded[/]")
    if failed:
        summary_parts.append(f"[red]✗ {failed} failed[/]")
    if running:
        summary_parts.append(f"[yellow]▶ {running} running[/]")

    lines.append(HLINE * min(max_width, 60))

    return "\n".join(lines)


def render_graph_simple(graph: TaskGraph, goal: str = "") -> str:
    """Simpler renderer that just shows topological order with indentation.

    This is a fallback for when the tree renderer produces confusing output
    on complex graphs.  It shows each node with its status and dependencies.
    """
    lines: list[str] = []
    header = f"Plan: {goal}" if goal else "Task Graph"
    success, failed, running, pending = graph.progress()
    total = graph.node_count
    lines.append(f"{header}  [{success + failed}/{total}]")
    lines.append("━" * 60)

    ordered = graph.topological_sort()

    # Compute indent level from dependency depth
    depths = _compute_depths(graph)

    for node in ordered:
        d = depths.get(node.id, 0)
        indent = "  " * (d + 1)
        icon = status_icon(node)
        deps = graph.get_dependencies(node.id)
        dep_str = ""
        if deps:
            dep_names = []
            for dep_id in deps:
                dep_node = graph.get_node(dep_id)
                if dep_node:
                    dep_names.append(dep_node.title[:30])
            if dep_names:
                arrow = "<-" if not _UNICODE else "←"
                dep_str = f"  {arrow} {', '.join(dep_names)}"

        line = f"{indent}{icon} {node.title}{dep_str}"
        if node.result and node.result.duration_ms > 0:
            line += f"  [{_fmt_duration(node.result.duration_ms)}]"
        if node.retry_count > 0:
            line += f"  (retry {node.retry_count})"
        lines.append(line)

    lines.append("━" * 60)

    # Legend
    lines.append(f"  {STATUS_ICONS['pending']} pending  {STATUS_ICONS['ready']} ready  "
                 f"{STATUS_ICONS['running']} running  {STATUS_ICONS['success']} done  "
                 f"{STATUS_ICONS['failed']} failed  {STATUS_ICONS['blocked']} blocked  "
                 f"{STATUS_ICONS['skipped']} skipped")
    return "\n".join(lines)


def render_graph_rich(graph: TaskGraph, goal: str = "") -> str:
    """Render with rich markup for color output in the CLI.

    Returns a string with rich markup tags that `rich.console.Console.print()`
    will interpret for colored output.
    """
    # Reuse the simple renderer, but wrap icons in rich color tags
    lines: list[str] = []
    header = f"Plan: {goal}" if goal else "Task Graph"
    success, failed, running, pending = graph.progress()
    total = graph.node_count
    lines.append(f"[bold]{header}[/]  [dim][{success + failed}/{total}][/]")
    lines.append("━" * 60)

    ordered = graph.topological_sort()
    depths = _compute_depths(graph)

    for node in ordered:
        d = depths.get(node.id, 0)
        indent = "  " * (d + 1)
        icon = status_icon(node)
        color = status_color(node)
        title = node.title

        if color:
            line = f"{indent}[{color}]{icon} {title}[/{color}]"
        else:
            line = f"{indent}{icon} {title}"

        if node.result and node.result.duration_ms > 0:
            line += f" [dim]{_fmt_duration(node.result.duration_ms)}[/]"
        if node.retry_count > 0:
            line += f" [yellow](retry {node.retry_count})[/]"
        if node.error:
            line += f" [red]{node.error[:60]}[/]"

        lines.append(line)

    lines.append("━" * 60)
    return "\n".join(lines)


def _compute_depths(graph: TaskGraph) -> dict[str, int]:
    """Compute the dependency depth of each node (0 = root, 1 = child of root, etc.)."""
    depths: dict[str, int] = {}
    ordered = graph.topological_sort()
    for node in ordered:
        deps = graph.get_dependencies(node.id)
        if not deps:
            depths[node.id] = 0
        else:
            depths[node.id] = 1 + max(depths.get(d, 0) for d in deps)
    return depths


def _fmt_duration(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60_000:
        return f"{ms / 1000:.1f}s"
    else:
        return f"{ms / 60_000:.1f}m"


def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Render a simple progress bar."""
    if total == 0:
        return "[          ] 0%"
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.0%}"
