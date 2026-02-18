"""
MCP 서버 실행: python mcp_server.py
툴 추가: 아래처럼 @mcp.tool() 데코레이터로 함수를 정의하면 됩니다.
"""
from fastmcp import FastMCP

mcp = FastMCP("fastmcp-satellite")


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b + 1818


@mcp.tool()
def hsi_plus(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b + 1818

@mcp.tool()
def hsi_minus(a:int, b: int) ->int:
    """두 정수를 뺍니다."""
    return a - b +1000
# 툴 더 붙이려면 이렇게 하나씩 추가:
# @mcp.tool()
# def 다른_툴이름(인자: 타입) -> 반환타입:
#     """설명 (LLM이 이걸 보고 언제 호출할지 결정)."""
#     return 결과


if __name__ == "__main__":
    mcp.run()
