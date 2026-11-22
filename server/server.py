
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools import TOOLS

mcp = FastMCP("CGPA Guider")


@mcp.tool
def test_connection() -> str:
    """Check whether the CGPA MCP server is working."""
    return "CGPA MCP server is working!"


for langchain_tool in TOOLS:
    mcp.tool(langchain_tool.func)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
