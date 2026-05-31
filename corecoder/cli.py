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
from .config import Config
from .mcp.client import MCPClient
from .prompt import system_prompt
from .context.session import save_session, load_session, list_sessions
from . import __version__

console = Console()
logger = logging.getLogger("corecoder")


class _StagedCliPresenter:
    """Small CLI presenter for the outer Think and inner Execute loop."""

    def __init__(self, console: Console):
        self.console = console
        self._active_stage: str | None = None
        self._in_execute = False
        self._react_announced = False

    def on_event(self, event: dict):
        event_type = str(event.get("type", ""))
        if event_type == "think_start":
            self._in_execute = False
            self._react_announced = False
            self.console.print("\n[bold cyan]Think[/] [dim]assessing current state and deciding the next bounded step...[/]")
            return
        if event_type == "think_complete":
            decision_type = event.get("decision_type")
            if decision_type == "stage_plan":
                stage = event.get("stage") or "unknown"
                objective = (event.get("objective") or "").strip()
                reason = (event.get("reason") or "").strip()
                self.console.print(
                    Panel(
                        f"[bold]{stage}[/bold]\n"
                        + (f"{objective}\n" if objective else "")
                        + (f"[dim]{reason}[/dim]" if reason else ""),
                        title="Think -> Next Stage",
                        border_style="cyan",
                    )
                )
            elif decision_type in {"final", "final_answer"}:
                reason = (event.get("reason") or "").strip()
                self.console.print(f"\n[bold green]Think[/] [dim]ready to answer directly[/]" + (f" [dim]({reason})[/]" if reason else ""))
            return
        if event_type == "execute_start":
            self._active_stage = str(event.get("stage") or "")
            self._in_execute = True
            self._react_announced = False
            allowed_tools = ", ".join(event.get("allowed_tools", [])) or "none"
            retrieval_mode = event.get("retrieval_mode") or "cached"
            retrieval_reason = (event.get("retrieval_reason") or "").strip()
            lines = [
                f"[bold]{self._active_stage}[/bold]",
                f"Tools: [cyan]{allowed_tools}[/]",
                f"Retrieval: [yellow]{retrieval_mode}[/]" + (f" [dim]({retrieval_reason})[/]" if retrieval_reason else ""),
            ]
            self.console.print(Panel("\n".join(lines), title="Execute", border_style="yellow"))
            return
        if event_type == "react_loop_start":
            self._react_announced = True
            allowed_tools = ", ".join(event.get("allowed_tools", [])) or "none"
            max_steps = event.get("max_tool_steps")
            self.console.print(
                f"[yellow]ReAct[/] [dim]entered local tool loop[/] "
                f"[dim](allowed: {allowed_tools}; max steps: {max_steps})[/]"
            )
            return
        if event_type == "execute_complete":
            stage = event.get("stage") or self._active_stage or "unknown"
            success = bool(event.get("success"))
            tool_count = event.get("tool_count", 0)
            observation_count = event.get("observation_count", 0)
            needs_replan = bool(event.get("needs_replan"))
            color = "green" if success and not needs_replan else "yellow"
            status = "completed" if success and not needs_replan else "needs rethink"
            self.console.print(
                f"[{color}]Execute[/] [bold]{stage}[/bold] [dim]{status}[/] "
                f"[dim](tools: {tool_count}, observations: {observation_count})[/]"
            )
            self._in_execute = False
            return
        if event_type == "evaluation":
            if event.get("needs_replan"):
                reason = (event.get("reason") or "").strip()
                self.console.print(f"[yellow]Evaluate[/] [dim]replan requested[/]" + (f" [dim]({reason})[/]" if reason else ""))
            return

    def on_tool(self, name: str, kwargs: dict):
        if self._in_execute and not self._react_announced:
            self.console.print("[yellow]ReAct[/] [dim]entered local tool loop[/]")
            self._react_announced = True
        prefix = f"[{self._active_stage}] " if self._active_stage else ""
        self.console.print(f"[dim]> {prefix}{name}({_brief(kwargs)})[/dim]")

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
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
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
    if args.debug:
        config.debug = True
    setup_logging(config.debug)
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
    mcp_client = MCPClient.from_config()
    await mcp_client.start_all()
    mcp_tools = mcp_client.all_tools()
    if mcp_tools:
        agent.tools.extend(mcp_tools)
        agent._system = system_prompt(agent.tools)
        console.print(f"[dim]MCP: {len(mcp_tools)} tools from {len(mcp_client.servers)} server(s)[/]")
    try:
        if args.resume:
            loaded = load_session(args.resume)
            if loaded:
                agent.state, loaded_model = loaded
                if not args.model:
                    agent.llm.model = loaded_model
                    config.model = loaded_model
                console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/]")
            else:
                console.print(f"[red]Session '{args.resume}' not found.[/]")
                sys.exit(1)
        if args.prompt:
            await _run_once(agent, args.prompt)
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
    presenter = _StagedCliPresenter(console)
    def on_token(tok):
        if tok.startswith("[think]"):
            return
        streamed.append(tok)
    from .context.orchestrator import ContextOrchestrator
    orch = ContextOrchestrator(working_dir=agent.working_dir, repo_index=agent.repo_index)
    state = await agent.run_staged(
        prompt,
        context_orchestrator=orch,
        on_token=on_token,
        on_tool=presenter.on_tool,
        on_event=presenter.on_event,
    )
    output = "".join(streamed) or state.final_answer or ""
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
                console.print(f"[yellow]Undone: {desc} ({len(agent.state.persistent_history)} messages remaining)[/]")
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
            from .context.compression import estimate_tokens
            before = estimate_tokens(agent.state.persistent_history)
            compressed = await agent.context.maybe_compress(agent.state.persistent_history, agent.llm)
            after = estimate_tokens(agent.state.persistent_history)
            if compressed:
                console.print(f"[green]Compressed: {before} -> {after} tokens ({len(agent.state.persistent_history)} messages)[/]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.state.persistent_history)} messages)[/]")
            continue
        if user_input == "/save":
            sid = save_session(agent.state, config.model)
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
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]}[/]")
            _show_help()
            continue
        streamed: list[str] = []
        presenter = _StagedCliPresenter(console)
        def on_token(tok):
            if tok.startswith("[think]"):
                return
            streamed.append(tok)
        try:
            from .context.orchestrator import ContextOrchestrator
            orch = ContextOrchestrator(working_dir=agent.working_dir, repo_index=agent.repo_index)
            state = await agent.run_staged(
                user_input,
                context_orchestrator=orch,
                on_token=on_token,
                on_tool=presenter.on_tool,
                on_event=presenter.on_event,
            )
            output = "".join(streamed) or state.final_answer or ""
            if output:
                console.print(Markdown(output))
            elif not streamed and not state.final_answer:
                pass
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            logger.exception("Error in agent loop")
            console.print(f"\n[red]Error: {e}[/]")

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _show_help():
    console.print(Panel(
        "[bold]Commands:[/]\n"
        "  /help          Show this help\n"
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
