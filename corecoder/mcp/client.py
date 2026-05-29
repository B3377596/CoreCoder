"""MCP (Model Context Protocol) client.

Connects to MCP servers over stdio, discovers their tools, and wraps them as
CoreCoder tools so the agent can use them transparently.

Configure servers in ``~/.corecoder/mcp.json``:

    {
      "servers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        },
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {"GITHUB_TOKEN": "ghp_xxx"}
        }
      }
    }
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path
from ..tools.base import Tool

logger = logging.getLogger("corecoder.mcp")

MCP_CONFIG_PATH = Path.home() / ".corecoder" / "mcp.json"

class MCPServer:
    """A single MCP server process connected via stdio (JSON-RPC 2.0)."""
    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.proc: asyncio.subprocess.Process | None = None
        self._tools: list[dict] = []
        self._id = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self):
        full_env = os.environ.copy()
        full_env.update(self.env)
        self.proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        caps = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "CoreCoder", "version": "0.3.0"},
        })
        logger.info("MCP [%s] initialized: %s", self.name,
                     caps.get("serverInfo", {}).get("name", "unknown"))
        await self._notify("initialized", {})
        result = await self._request("tools/list", {})
        self._tools = result.get("tools", [])
        logger.info("MCP [%s]: %d tools discovered", self.name, len(self._tools))

    async def close(self):
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, Exception):
                self.proc.kill()
                await self.proc.wait()

    # ------------------------------------------------------------------
    # tool calling
    # ------------------------------------------------------------------
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "resource":
                resource = item.get("resource", {})
                parts.append(f"[resource: {resource.get('uri', '*')}]")
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts) or "(tool returned no text)"

    @property
    def tools(self) -> list[dict]:
        return self._tools

    # ------------------------------------------------------------------
    # JSON-RPC over stdio
    # ------------------------------------------------------------------
    async def _request(self, method: str, params: dict) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        async with self._lock:
            return await self._send(msg)

    async def _notify(self, method: str, params: dict):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self._lock:
            line = json.dumps(msg) + "\n"
            self.proc.stdin.write(line.encode())
            await self.proc.stdin.drain()

    async def _send(self, msg: dict) -> dict:
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        await self.proc.stdin.drain()
        response_line = await self.proc.stdout.readline()
        if not response_line:
            raise ConnectionError(f"MCP server '{self.name}' closed unexpectedly")
        response = json.loads(response_line.decode())
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error [{err.get('code', -1)}]: {err.get('message', 'unknown')}")
        return response.get("result", {})

class MCPTool(Tool):
    """A CoreCoder tool that delegates to an MCP server."""
    def __init__(self, server: MCPServer, tool_def: dict):
        self._server = server
        self._tool_def = tool_def
        self.name = tool_def["name"]
        self.description = tool_def.get("description", f"MCP tool from {server.name}")
        self.parameters = tool_def.get("inputSchema", {
            "type": "object",
            "properties": {},
        })

    async def execute(self, **kwargs) -> str:
        try:
            return await self._server.call_tool(self.name, kwargs)
        except Exception as e:
            return f"MCP tool error ({self.name} @ {self._server.name}): {e}"

class MCPClient:
    """Manages a collection of MCP servers and provides their tools."""
    def __init__(self):
        self.servers: list[MCPServer] = []

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "MCPClient":
        path = config_path or MCP_CONFIG_PATH
        client = cls()
        if not path.exists():
            logger.debug("No MCP config at %s", path)
            return client
        try:
            config = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load MCP config: %s", e)
            return client
        servers_cfg = config.get("servers", {}) if isinstance(config, dict) else {}
        for name, cfg in servers_cfg.items():
            if not isinstance(cfg, dict) or "command" not in cfg:
                logger.warning("Skipping invalid MCP server entry: %s", name)
                continue
            server = MCPServer(
                name=name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
            )
            client.servers.append(server)
            logger.debug("Loaded MCP server config: %s (%s)", name, cfg["command"])
        return client

    async def start_all(self):
        for s in self.servers:
            try:
                await s.start()
            except Exception as e:
                logger.warning("Failed to start MCP server [%s]: %s", s.name, e)

    async def close_all(self):
        for s in self.servers:
            try:
                await s.close()
            except Exception as e:
                logger.warning("Error closing MCP server [%s]: %s", s.name, e)

    def all_tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        for s in self.servers:
            for td in s.tools:
                tools.append(MCPTool(s, td))
        return tools
