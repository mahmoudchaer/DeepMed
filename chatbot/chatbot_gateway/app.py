# chatbot/chatbot_gateway/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import os
import httpx

EMB_URL       = os.getenv("EMBEDDING_URL", "http://embedding_service:5201")
VEC_URL       = os.getenv("VECTOR_URL",    "http://vector_search_service:5202")
LLM_URL       = os.getenv("LLM_URL",       "http://llm_generator_service:5203")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant for DeepMed. Only answer questions about the platform or medical AI. If off-topic, politely decline."
)

app = FastAPI(title="DeepMed Chatbot Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatReq(BaseModel):
    user_id: str = Field(..., description="Authenticated user identifier")
    history: List[Message] = Field(default_factory=list)
    message: str = Field(..., description="New user message")

class ChatRes(BaseModel):
    reply: str

@app.post("/chatbot/query", response_model=ChatRes)
async def chatbot_query(req: ChatReq):
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1) Embed
        try:
            e = await client.post(f"{EMB_URL}/embedding/generate", json={"text": req.message})
            e.raise_for_status()
            embedding = e.json()["embedding"]
        except Exception as ex:
            raise HTTPException(502, f"Embedding error: {ex}")

        # 2) Vector search
        try:
            v = await client.post(
                f"{VEC_URL}/vector/search",
                json={"embedding": embedding, "k": 4}
            )
            v.raise_for_status()
            contexts = v.json().get("documents", [])
        except Exception as ex:
            raise HTTPException(502, f"Vector search error: {ex}")

        # 3) Build prompt
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += [m.dict() for m in req.history]
        for c in contexts:
            msgs.append({"role": "system", "content": c})
        msgs.append({"role": "user", "content": req.message})

        # 4) LLM
        try:
            l = await client.post(f"{LLM_URL}/llm/generate", json={"messages": msgs})
            l.raise_for_status()
            reply = l.json().get("reply", "")
        except Exception as ex:
            raise HTTPException(502, f"LLM error: {ex}")

        return {"reply": reply}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5204, reload=True)
