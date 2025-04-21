# chatbot/llm_generator_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.2

class LLMResponse(BaseModel):
    reply: str

@app.post("/llm/generate", response_model=LLMResponse)
async def generate(llm_req: LLMRequest):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m.role, "content": m.content} for m in llm_req.messages],
            max_tokens=llm_req.max_tokens,
            temperature=llm_req.temperature
        )
        return LLMResponse(reply=resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5203, reload=True)
