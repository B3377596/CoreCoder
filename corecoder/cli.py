"""Interactive REPL - the user-facing terminal interface."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import nest_asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.logging import RichHandler
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm.client import LLM, LiteLLM
from .llm.types import LLMResponse
from .config import Config
from .mcp.client import MCPClient
from .prompt import system_prompt
from .history.session import save_session, load_session, list_sessions
from . import __version__
from .orchestration.viz import render_graph_rich, status_icon
from .orchestration.orchestrator import Orchestrator, OrchestratorConfig
from .orchestration.engine.planner import LLMPlanner

console = Console()
logger = logging.getLogger("corecoder")


# ------------------------------------------------------------------
# argument parsing
# ------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-P", "--plan", metavar="GOAL", help="One-shot orchestrated plan mode")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


# ------------------------------------------------------------------
# logging setup
# ------------------------------------------------------------------


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.WARNING
    handler = RichHandler(console=console, rich_tracebacks=True, show_time=True)
    handler.setLevel(level)
    # also configure root logger for library noise
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    # our own loggers at requested level
    for name in ("corecoder", "corecoder.agent", "corecoder.context",
                 "corecoder.tools", "corecoder.mcp"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers = [handler]
        lg.propagate = False


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------


def main():
    """Sync entry point for console_scripts."""
    nest_asyncio.apply()

    # Suppress noisy "Task exception was never retrieved" for
    # KeyboardInterrupt inside prompt_toolkit's nested event loop.
    def _exc_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, KeyboardInterrupt):
            return
        loop.default_exception_handler(context)
    asyncio.get_event_loop().set_exception_handler(_exc_handler)

    args = _parse_args()
    config = Config.from_env()

    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    setup_logging(args.debug)

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    asyncio.run(_async_main(agent, config, args))


async def _async_main(agent: Agent, config: Config, args):
    """Async body - runs inside asyncio.run()."""

    # --- MCP: start servers and register their tools ---
    mcp_client = MCPClient.from_config()
    await mcp_client.start_all()
    mcp_tools = mcp_client.all_tools()
    if mcp_tools:
        agent.tools.extend(mcp_tools)
        agent._system = system_prompt(agent.tools, agent.repo_index.summary)
        console.print(f"[dim]MCP: {len(mcp_tools)} tools from {len(mcp_client.servers)} server(s)[/]")

    try:
        # resume saved session
        if args.resume:
            loaded = load_session(args.resume)
            if loaded:
                agent.messages, loaded_model = loaded
                if not args.model:
                    agent.llm.model = loaded_model
                    config.model = loaded_model
                console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/]")
            else:
                console.print(f"[red]Session '{args.resume}' not found.[/]")
                sys.exit(1)

        if args.prompt:
            await _run_once(agent, args.prompt)
        elif args.plan:
            await _run_plan(agent, args.plan)
        else:
            await _repl(agent, config)
    finally:
        await mcp_client.close_all()


# ------------------------------------------------------------------
# one-shot mode
# ------------------------------------------------------------------


async def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    streamed: list[str] = []

    def on_token(tok):
        streamed.append(tok)

    def on_tool(name, kwargs):
        console.print(f"[dim]> {name}({_brief(kwargs)})[/dim]")

    response = await agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    output = "".join(streamed) or response
    if output:
        console.print(Markdown(output))


# ------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------


async def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")
    
    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in slash commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/]")
            continue
        if user_input == "/undo":
            desc = agent.undo()
            if desc:
                console.print(f"[yellow]Undone: {desc} ({len(agent.messages)} messages remaining)[/]")
            else:
                console.print("[dim]Nothing to undo.[/]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/] prompt + [cyan]{c}[/] completion = [bold]{p+c}[/] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/]")
            continue
        if user_input == "/compact":
            from .history.compression import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = await agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} 鈫?{after} tokens ({len(agent.messages)} messages)[/]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue
        if user_input == "/diff":
            files = agent.changed_files
            if not files:
                console.print("[dim]No files modified this turn.[/]")
            else:
                console.print(f"[bold]Files modified this turn ({len(files)}):[/]")
                for f in sorted(files):
                    console.print(f"  [cyan]{f}[/]")
                diff = agent.last_diff
                if diff and diff != "(no diff available)":
                    console.print(f"\n[dim]{diff[:2000]}[/]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue
        if user_input.startswith("/plan "):
            goal = user_input[6:].strip()
            if goal:
                await _run_plan(agent, goal)
            else:
                console.print("[yellow]Usage: /plan <goal description>[/]")
            continue

        elif user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]}[/]")
            _show_help();
            continue
        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)

        def on_tool(name, kwargs):
            console.print(f"[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = await agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            # render the complete response with proper markdown formatting
            output = "".join(streamed) or response
            if output:
                console.print(Markdown(output))
            elif not streamed and not response:
                pass  # nothing to show
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            logger.exception("Error in agent loop")
            console.print(f"\n[red]Error: {e}[/]")


# ------------------------------------------------------------------
# plan mode (orchestrated DAG execution)
# ------------------------------------------------------------------


async def _run_plan(agent: Agent, goal: str):
    """Execute a goal through the DAG orchestration pipeline.

    1. LLMPlanner decomposes the goal into a task graph
    2. The graph is displayed to the user
    3. Each task executes via the agent's ReAct loop
    4. Progress is shown in real-time
    """
    from .orchestration.orchestrator import Orchestrator, OrchestratorConfig
    from .orchestration.engine.planner import LLMPlanner
    from .orchestration.viz import render_graph_rich

    console.print(f"\n[bold blue]Planning:[/] {goal}")

    # ---- Phase 1: Plan (LLM decomposes goal into task graph) ----
    # Show a spinner with elapsed time during LLM planning (can take 30s-2m
    # depending on model and prompt length)
    import time as _time
    planning_start = _time.time()

    async def llm_call(messages: list[dict]) -> LLMResponse:
        return await agent.llm.chat(messages=messages, tools=None)

    planner = LLMPlanner(llm_call=llm_call, model=agent.llm.model)

    async def _plan_with_progress():
        nonlocal plan_result
        plan_result = await planner.aplan(goal)

    plan_result = None
    plan_error: Exception | None = None
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    plan_task = asyncio.create_task(_plan_with_progress())
    planning_timeout_s = 180  # 3 minute timeout for planning
    try:
        while not plan_task.done():
            elapsed = _time.time() - planning_start
            spinner = spinner_chars[i % len(spinner_chars)]
            # Show elapsed; warn if taking unusually long (>30s)
            hint = " [yellow](LLM may be reasoning...)[/]" if elapsed > 30 else ""
            sys.stdout.write(f"\r  {spinner} 规划中... [dim]{elapsed:.0f}s[/]{hint}")
            sys.stdout.flush()
            i += 1
            if elapsed > planning_timeout_s:
                plan_task.cancel()
                raise TimeoutError(f"Planning timed out after {planning_timeout_s}s")
            await asyncio.sleep(0.15)
        await plan_task
        if plan_task.exception():
            plan_error = plan_task.exception()
    except (asyncio.CancelledError, KeyboardInterrupt):
        plan_task.cancel()
        raise
    except TimeoutError:
        plan_error = TimeoutError(f"Planning timed out after {planning_timeout_s}s")
    except Exception as e:
        plan_error = e
    finally:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    if plan_error:
        console.print(f"[red]Planning failed: {plan_error}[/]")
        console.print("[yellow]Falling back to direct execution.[/]\n")
        await _run_once(agent, goal)
        return

    plan = plan_result
    plan_elapsed = _time.time() - planning_start
    console.print(f"  [green]Done[/] [dim]({plan_elapsed:.1f}s, {plan.graph.node_count} tasks)[/]\n")

    if plan.graph.node_count == 0:
        console.print("[yellow]Planner produced an empty plan.  Falling back to direct execution.[/]\n")
        await _run_once(agent, goal)
        return

    # ---- Phase 2: Show the plan ----
    console.print(render_graph_rich(plan.graph, goal))
    console.print()

    # ---- Phase 3: Execute with progress ----
    # Track which task is currently executing (for tool call context in parallel mode)
    _current_task_title: str = ""

    def on_tool(name: str, kwargs: dict):
        """Print each tool invocation with the task context."""
        brief = _brief(kwargs, 120)
        prefix = f"    [{_current_task_title[:20]}]" if _current_task_title else "    "
        console.print(f"{prefix} [dim]> {name}({brief})[/dim]")

    def on_progress(task_node, event: str):
        """Print task status transitions with timing and verification details."""
        nonlocal _current_task_title
        if event == "running":
            _current_task_title = task_node.title
        icon = status_icon(task_node)
        color_map = {
            "running": "bold yellow",
            "success": "green",
            "retry": "yellow",
            "skipped": "red",
        }
        color = color_map.get(event, "")
        msg = f"  {icon} {task_node.title}"
        if task_node.result and task_node.result.duration_ms > 0:
            ms = task_node.result.duration_ms
            if ms < 1000:
                msg += f" [{ms:.0f}ms]"
            else:
                msg += f" [{ms / 1000:.1f}s]"
            if task_node.result.tool_calls_made > 0:
                msg += f" ({task_node.result.tool_calls_made} tools)"
        if event == "retry":
            msg += f" (retry {task_node.retry_count}/{task_node.retry_policy.max_retries})"
        if color:
            console.print(f"[{color}]{msg}[/{color}]")
        else:
            console.print(msg)

        # Show verification failure details so user can debug
        if event in ("retry", "skipped") and task_node.verification:
            v = task_node.verification
            if v.failures:
                for f in v.failures[:3]:  # Show at most 3 failures
                    console.print(f"    [red]verify: {f[:200]}[/]")
            if v.warnings:
                for w in v.warnings[:2]:
                    console.print(f"    [yellow]warn: {w[:200]}[/]")

        sys.stdout.flush()

    orch_config = OrchestratorConfig(
        goal=goal,
        continue_on_failure=True,
        auto_persist=True,
        max_rounds_per_task=15,  # each orchestrated task is focused
        parallel=True,           # run independent tasks concurrently
        max_parallel=4,
        on_tool_callback=on_tool,
    )
    orch = Orchestrator(orch_config)
    orch.set_planner(planner)
    orch.set_agent(agent.chat, agent_instance=agent)  # pass full Agent for cloning
    orch.on_progress(on_progress)

    console.print("[bold]Executing plan...[/]\n")
    result = await orch.run(goal, plan=plan)

    # ---- Phase 4: Report ----
    console.print()
    if result.success:
        console.print(
            f"[bold green]Plan completed:[/] "
            f"{result.tasks_succeeded}/{result.tasks_total} tasks succeeded "
            f"in {result.total_duration_ms:.0f}ms"
        )
    else:
        console.print(
            f"[bold red]Plan finished with failures:[/] "
            f"{result.tasks_succeeded} succeeded, {result.tasks_failed} failed, "
            f"{result.tasks_skipped} skipped"
        )
    if result.replans_used:
        console.print(f"[yellow]Replans used: {result.replans_used}[/]")

    # Show final graph state
    if result.graph:
        console.print()
        console.print(render_graph_rich(result.graph, goal))


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/]\n"
        "  /help          Show this help\n"
        "  /plan <goal>   Execute a goal through DAG orchestration\n"
        "                 (LLM plans → shows task graph → executes with progress)\n"
        "  /reset         Clear conversation history\n"
        "  /undo          Undo last user turn\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 200) -> str:
    s = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
