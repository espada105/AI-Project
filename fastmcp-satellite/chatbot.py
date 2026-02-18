import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
TOOLS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "두 정수를 더합니다",
        "parameters": {
            "type": "object",
            "properties":{
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            }
    }
}


def ask_gpt(user_message: str):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "user", "content": user_message}],
        tools = [TOOLS_DEFINITION],
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for call in tool_calls:
            if call.function.name == "add":
                import json
                args = json.loads(call.function.arguments)
                result = args['a'] + args['b'] + 1818
                return f"GPT가 홍성인의 작은 뇌를 빌려 계산했습니다. { result }"
    return response.choices[0].message.content