"""
FastMCP 학습용 - 최소 예제.
실행: python main.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-first-mcp")


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더합니다."""
    return a + b


if __name__ == "__main__":
    mcp.run()
