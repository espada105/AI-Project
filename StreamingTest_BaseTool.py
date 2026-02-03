"""
Generated from `StreamingTest_BaseTool.ipynb`.

Notes
- This script loads API key from `.env` (expects `GPT_API_KEY` or `OPENAI_API_KEY`).
- Run: `python StreamingTest_BaseTool.py`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field


def load_env() -> None:
    """Load .env from cwd or parent, and map GPT_API_KEY -> OPENAI_API_KEY."""
    for d in (Path.cwd(), Path.cwd().parent):
        env_file = d / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
    else:
        load_dotenv()

    key = (os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key


class GetWeatherInput(BaseModel):
    city: str = Field(description="날씨 조회할 도시 이름")


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "도시 이름을 받아 해당 도시의 날씨를 반환하는 도구"
    args_schema: type[BaseModel] = GetWeatherInput
    response_format: str = "content_and_artifact"

    def _run(
        self,
        city: str,
        run_manager: Optional[Any] = None,
    ) -> tuple[str, Any]:
        writer = get_stream_writer()
        writer(f"{city}의 기상 관측 위성에 접속 중")
        writer(f"{city}의 기상 데이터 수신 완료")

        content = f"{city}의 기상 상태는 맑음, 25도임 ㅇㅇ"
        artifact = {"city": city, "temperature": 25, "codition": "맑음"}
        return content, artifact

    async def _arun(
        self,
        city: str,
        run_manager: Optional[Any] = None,
    ) -> tuple[str, Any]:
        writer = get_stream_writer()
        writer(f"{city}의 기상 관측 위성에 접속 중")
        writer(f"{city}의 기상 데이터 수신 완료")

        content = f"{city}의 기상 상태는 맑음, 25도임 ㅇㅇ"
        artifact = {"city": city, "temperature": 25, "condition": "맑음"}
        return content, artifact


def main() -> None:
    load_env()

    # Tool instance & agent
    get_weather = GetWeatherTool()
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_agent(llm, tools=[get_weather])

    # test input data
    input_data = {"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]}

    # mode 1: updates
    print("[updates mode]")
    for chunk in agent.stream(input_data, stream_mode="updates"):
        for node_name, data in chunk.items():
            print(f"\n node: {node_name}")
            if "messages" in data:
                last_msg = data["messages"][-1]
                print(f" messge type: {type(last_msg).__name__}")
                if hasattr(last_msg, "content"):
                    print(f"content: {last_msg.content[:100]}...")

    # mode 2: messages
    print("messages mode")
    for token, metadata in agent.stream(input_data, stream_mode="messages"):
        if token.content:
            for block in token.content_blocks:
                if block.get("text"):
                    print(block["text"], end="", flush=True)
    print()

    # mode3: custom
    print("custom mode")
    for chunk in agent.stream(input_data, stream_mode="custom"):
        print(f" [Log] {chunk}")

    # Final 통합
    print("=== Mode 4: 여러 모드 동시 ===")
    for mode, data in agent.stream(input_data, stream_mode=["updates", "custom", "messages"]):
        if mode == "updates":
            node = list(data.keys())[0]
            print(f"\n[업데이트] {node} 단계 완료")
        elif mode == "custom":
            print(f"[커스텀] {data}")
        elif mode == "messages":
            token, meta = data
            if token.content:
                text = "".join([b.get("text", "") for b in token.content_blocks])
                print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

