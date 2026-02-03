"""
Generated from `StreamigTest.ipynb`.

Important
- The notebook currently references `agent`/`input_data` in later cells.
  If you want this script to run end-to-end, you must define `agent` before
  running the streaming sections (see `StreamingTest_BaseTool.py` for a working setup).
"""

from __future__ import annotations


def mode_1_updates(agent, input_data) -> None:
    print(" [updates Mode start]")
    for chunk in agent.stream(input_data, stream_mode="updates"):
        for node_name, data in chunk.items():
            print(f"현재 단계 {node_name}")
            last_msg = data["messages"][-1]
            print(f"summary {type(last_msg).__name__}")


def mode_2_messages(agent, input_data) -> None:
    print(" [messages Mode start]")
    for token, metadata in agent.stream(input_data, stream_mode="messages"):
        if token.content:
            for block in token.content_blocks:
                if block.get("text"):
                    print(block["text"], end="|", flush=True)


def mode_3_custom(agent, input_data) -> None:
    print("[custom Mode start]")
    for chunk in agent.stream(input_data, stream_mode="custom"):
        print(f" tool Log : {chunk}")
