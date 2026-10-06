import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langfuse import observe, get_client

_session: ClientSession | None = None


@asynccontextmanager
async def connect(server_script: str = "mcp_server.py"):
    """Start the MCP server as a subprocess and open a session over stdio."""
    global _session
    params = StdioServerParameters(command=sys.executable, args=[server_script])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _session = session
            try:
                yield session
            finally:
                _session = None

@observe(name="mcp tools/list", capture_input=False, capture_output=False)
async def list_tool_schemas() -> list[dict]:
    """Sends tools/list and converts the result to the format the LLM expects."""
    result = await _session.list_tools()
    schemas = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in result.tools
    ]
    get_client().update_current_span(
        input={"jsonrpc": "2.0", "method": "tools/list"},
        output={"jsonrpc": "2.0", "result": {"tools": schemas}},
    )
    return schemas


@observe(name="mcp tools/call", capture_input=False, capture_output=False)
async def call_tool(name: str, args: dict) -> tuple[str, bool]:
    """Sends tools/call. Returns (text_output, is_error)."""
    result = await _session.call_tool(name, args)
    text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
    is_error = bool(result.isError)
    get_client().update_current_span(
        input={"jsonrpc": "2.0", "method": "tools/call", "params": {"name": name, "arguments": args}},
        output={"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": text}], "isError": is_error}},
    )
    return text, is_error