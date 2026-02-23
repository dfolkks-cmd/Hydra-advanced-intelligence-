"""
integrations/mcp_client.py
───────────────────────────
Unified MCP client — wires every external MCP server into LangChain-compatible
tool objects that LangGraph specialist agents can call via bind_tools().

SERVERS INTEGRATED:
  market_intelligence  — Hosted market data server (our own, on Alpic)
  memory_server        — Hosted Qdrant semantic memory (our own, on Alpic)
  coingecko_official   — Official CoinGecko MCP (public, no key required)
  prometheus           — Prometheus metrics queries (self-hosted)
  github               — GitHub repo access (official Anthropic reference)

DESIGN:
  Each server is registered in MCP_SERVER_CONFIG. The MCPClientManager
  connects to all of them at startup, loads their tools, and returns them
  as named groups. Specialist agents import the groups they need:

    from integrations.mcp_client import get_market_tools, get_memory_tools

  All tools are standard LangChain BaseTool objects — they slot directly
  into bind_tools() and tool_node() without any extra plumbing.

  The manager gracefully degrades: if a server is unreachable at startup,
  its tools are skipped and a warning is logged. The system continues with
  whatever servers are available.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Server configuration ─────────────────────────────────────────────────────
# Each entry describes one MCP server and how to connect to it.
# transport: "http" = streamable-http over network (for Alpic-hosted servers)
#            "stdio" = subprocess (for locally-installed reference servers)

MCP_SERVER_CONFIG: dict[str, dict[str, Any]] = {

    # ── Our own Alpic-hosted servers ─────────────────────────────────────────
    "market_intelligence": {
        "transport": "http",
        "url": os.getenv(
            "MARKET_INTELLIGENCE_URL",
            "http://localhost:8001/mcp"      # local dev default
        ),
        "headers": {
            "x-api-key": os.getenv("HYDRA_API_KEY", "")
        },
        "tool_group": "market",
        "description": "Hydra market intelligence: price, OHLCV, technicals, news, on-chain",
    },

    "memory_server": {
        "transport": "http",
        "url": os.getenv(
            "MEMORY_SERVER_URL",
            "http://localhost:8002/mcp"
        ),
        "headers": {
            "x-api-key": os.getenv("HYDRA_API_KEY", "")
        },
        "tool_group": "memory",
        "description": "Qdrant-backed semantic memory: store/search analyses, rejections, ThoughtNodes",
    },

    # ── Official CoinGecko MCP (public endpoint) ─────────────────────────────
    "coingecko_official": {
        "transport": "http",
        "url": "https://mcp.api.coingecko.com/mcp",
        "headers": {},
        "tool_group": "market",
        "description": "Official CoinGecko MCP: 15k+ coins, DEX data, trending, categories",
    },

    # ── Prometheus (self-hosted, read-only metrics queries) ──────────────────
    "prometheus": {
        "transport": "http",
        "url": os.getenv(
            "PROMETHEUS_MCP_URL",
            "http://localhost:9090/mcp"
        ),
        "headers": {},
        "tool_group": "observability",
        "description": "Prometheus metrics: agent latency, GoT survival rates, queue depth",
    },

    # ── GitHub reference server (stdio — runs as subprocess) ─────────────────
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-github",
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", ""),
        },
        "tool_group": "meta",
        "description": "GitHub: read repo issues, commits, code — for system self-awareness",
    },
}


# ── MCPClientManager ─────────────────────────────────────────────────────────

class MCPClientManager:
    """
    Manages connections to all configured MCP servers and provides
    LangChain-compatible tool objects grouped by function.

    Usage:
        manager = MCPClientManager()
        await manager.initialise()
        market_tools = manager.get_tools("market")
        memory_tools = manager.get_tools("memory")
        await manager.close()

    Or as an async context manager:
        async with MCPClientManager() as manager:
            tools = manager.get_tools("market")
    """

    def __init__(self, config: dict[str, dict] | None = None):
        self._config   = config or MCP_SERVER_CONFIG
        self._tools:   dict[str, list] = {}    # group_name → [LangChain tools]
        self._clients: list[Any]       = []    # for cleanup

    async def initialise(self) -> None:
        """Connect to all configured servers and load their tools."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            raise RuntimeError(
                "langchain-mcp-adapters not installed. "
                "Run: pip install langchain-mcp-adapters"
            )

        # Build the config format MultiServerMCPClient expects
        client_config: dict[str, dict] = {}

        for name, cfg in self._config.items():
            transport = cfg.get("transport", "http")
            if transport == "http":
                client_config[name] = {
                    "transport": "streamable_http",
                    "url":       cfg["url"],
                    "headers":   cfg.get("headers", {}),
                }
            elif transport == "stdio":
                client_config[name] = {
                    "transport": "stdio",
                    "command":   cfg["command"],
                    "args":      cfg.get("args", []),
                    "env":       cfg.get("env", {}),
                }

        try:
            self._mcp_client = MultiServerMCPClient(client_config)
            await self._mcp_client.__aenter__()

            all_tools = await self._mcp_client.get_tools()

            # Group tools by which server they came from
            for tool in all_tools:
                # MultiServerMCPClient prefixes tool names with server name
                # e.g. "market_intelligence__get_price" → group "market"
                server_name = self._infer_server(tool.name)
                group       = self._config.get(server_name, {}).get("tool_group", "misc")
                if group not in self._tools:
                    self._tools[group] = []
                self._tools[group].append(tool)
                logger.info(f"  ✓ Loaded tool: {tool.name} → group:{group}")

            self._clients.append(self._mcp_client)
            logger.info(f"MCPClientManager: loaded {sum(len(v) for v in self._tools.values())} tools across {len(self._tools)} groups")

        except Exception as e:
            logger.warning(f"MCPClientManager: partial init failure — {e}. Using available tools only.")

    def _infer_server(self, tool_name: str) -> str:
        """Infer which server owns a tool from its prefixed name."""
        for server_name in self._config:
            if tool_name.startswith(server_name):
                return server_name
        return "unknown"

    def get_tools(self, group: str) -> list:
        """Return all tools belonging to the named group."""
        return self._tools.get(group, [])

    def get_all_tools(self) -> list:
        """Return every tool across all groups."""
        return [t for tools in self._tools.values() for t in tools]

    def tool_summary(self) -> dict[str, list[str]]:
        """Return a {group: [tool_names]} summary for logging/debugging."""
        return {
            group: [t.name for t in tools]
            for group, tools in self._tools.items()
        }

    async def close(self) -> None:
        for client in self._clients:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass

    async def __aenter__(self):
        await self.initialise()
        return self

    async def __aexit__(self, *args):
        await self.close()


# ── Module-level singleton + convenience accessors ───────────────────────────
# The LangGraph graph runner initialises this once at startup and shares it
# across all agent nodes. Individual nodes call get_market_tools() etc.

_manager: MCPClientManager | None = None


async def init_mcp_client(config: dict | None = None) -> MCPClientManager:
    """
    Initialise the module-level MCP client singleton.
    Call once at application startup (e.g. in the FastAPI lifespan).
    """
    global _manager
    _manager = MCPClientManager(config)
    await _manager.initialise()
    logger.info("MCP client singleton initialised")
    logger.info(f"Available tool groups: {list(_manager._tools.keys())}")
    return _manager


def get_market_tools() -> list:
    """Market data tools: price, OHLCV, technicals, news, on-chain."""
    if _manager is None:
        logger.warning("MCP client not initialised — returning empty market tools")
        return []
    return _manager.get_tools("market")


def get_memory_tools() -> list:
    """Memory tools: store/search analyses, rejections, ThoughtNodes."""
    if _manager is None:
        return []
    return _manager.get_tools("memory")


def get_observability_tools() -> list:
    """Prometheus metrics query tools."""
    if _manager is None:
        return []
    return _manager.get_tools("observability")


def get_meta_tools() -> list:
    """Meta tools: GitHub, system introspection."""
    if _manager is None:
        return []
    return _manager.get_tools("meta")


async def close_mcp_client() -> None:
    """Graceful shutdown — call in FastAPI lifespan cleanup."""
    global _manager
    if _manager:
        await _manager.close()
        _manager = None
