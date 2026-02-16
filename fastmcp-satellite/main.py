from fastapi import FastAPI
from pydantic import BaseModel
import chatbot

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    response = chatbot.ask_gpt(req.message)
    return {"reply": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)