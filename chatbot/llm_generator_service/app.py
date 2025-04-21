# chatbot/llm_generator_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OPENAI_API_KEY environment variable")
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized")

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
        logger.info(f"Generating response with {len(llm_req.messages)} messages")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m.role, "content": m.content} for m in llm_req.messages],
            max_tokens=llm_req.max_tokens,
            temperature=llm_req.temperature
        )
        reply = resp.choices[0].message.content
        logger.info(f"Successfully generated reply: {reply[:50]}...")
        return LLMResponse(reply=reply)
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm_generator"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LLM generator service on port 5203")
    uvicorn.run("app:app", host="0.0.0.0", port=5203, reload=True)
