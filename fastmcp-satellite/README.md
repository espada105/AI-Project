# FastMCP 처음부터 학습

## MCP 툴 붙이는 방법

1. **mcp_server.py** 를 연다.
2. 아래처럼 **함수 위에 `@mcp.tool()`** 를 붙이면 그 함수가 MCP 툴로 등록된다.

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""   # ← 이 설명은 LLM이 툴 선택할 때 참고함
    return a + b
```

3. 새 툴을 더 붙이려면 같은 방식으로 함수 정의 후 `@mcp.tool()` 만 붙이면 된다.

## 실행

```bash
pip install fastmcp
python mcp_server.py   # MCP 서버 (툴 제공)
python main.py         # FastAPI 채팅 앱 (별도)
```
