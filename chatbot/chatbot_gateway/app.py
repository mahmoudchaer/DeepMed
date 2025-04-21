# chatbot/chatbot_gateway/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import os
import httpx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMB_URL       = os.getenv("EMBEDDING_URL", "http://embedding_service:5201")
VEC_URL       = os.getenv("VECTOR_URL",    "http://vector_search_service:5202")
LLM_URL       = os.getenv("LLM_URL",       "http://llm_generator_service:5203")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant for DeepMed. Only answer questions about the platform or medical AI. If off-topic, politely decline."
)

# Guardrail for responses
GUARDRAIL_DISCLAIMER = os.getenv(
    "GUARDRAIL_DISCLAIMER",
    "You are an AI assistant for DeepMed, a no-code AI platform for medical professionals. Your purpose is to help users understand and use the DeepMed platform — including how to upload data, configure models, interpret results, and navigate the interface. You are also allowed to explain relevant AI concepts **as they apply to DeepMed**, such as why small datasets might lead to poor accuracy, or what a classification model does. However, you must not act like a general-purpose AI tutor. Only explain AI in the context of how DeepMed uses it. You must never provide medical advice, diagnoses, or treatment suggestions under any circumstances. If a user asks for such help, respond with: \"I’m not qualified to answer that. Please consult a healthcare professional.\" You must also not answer any questions unrelated to DeepMed. Even if the user says it's relevant or claims it will help them use the site, do not respond unless you know it is within the scope of DeepMed. You know exactly what DeepMed does. If something is not part of the platform, clearly say: That is not part of DeepMed’s functionality. Always stay focused on helping users safely and effectively use DeepMed and its built-in AI features."
)

# Whether to add the disclaimer to every response
ADD_DISCLAIMER = os.getenv("ADD_DISCLAIMER", "true").lower() in ("true", "1", "yes")

# Log service URLs for debugging
logger.info(f"EMB_URL: {EMB_URL}")
logger.info(f"VEC_URL: {VEC_URL}")
logger.info(f"LLM_URL: {LLM_URL}")

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
    logger.info(f"Processing request for user: {req.user_id}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) Embed
        try:
            logger.info(f"Calling embedding service at: {EMB_URL}")
            e = await client.post(f"{EMB_URL}/embedding/generate", json={"text": req.message})
            e.raise_for_status()
            embedding = e.json()["embedding"]
            logger.info("Successfully received embedding")
        except Exception as ex:
            logger.error(f"Embedding error: {ex}")
            raise HTTPException(502, f"Embedding error: {ex}")

        # 2) Vector search
        try:
            logger.info(f"Calling vector search service at: {VEC_URL}")
            v = await client.post(
                f"{VEC_URL}/vector/search",
                json={"embedding": embedding, "k": 4}
            )
            v.raise_for_status()
            contexts = v.json().get("documents", [])
            logger.info(f"Retrieved {len(contexts)} context documents")
        except Exception as ex:
            logger.error(f"Vector search error: {ex}")
            raise HTTPException(502, f"Vector search error: {ex}")

        # 3) Build prompt
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += [m.dict() for m in req.history]
        for c in contexts:
            msgs.append({"role": "system", "content": c})
        msgs.append({"role": "user", "content": req.message})

        # 4) LLM
        try:
            logger.info(f"Calling LLM service at: {LLM_URL}")
            l = await client.post(f"{LLM_URL}/llm/generate", json={"messages": msgs})
            l.raise_for_status()
            reply = l.json().get("reply", "")
            logger.info("Successfully generated reply")
        except Exception as ex:
            logger.error(f"LLM error: {ex}")
            raise HTTPException(502, f"LLM error: {ex}")

        # 5) Apply response guardrail if configured
        if ADD_DISCLAIMER:
            reply = f"{reply}\n\n{GUARDRAIL_DISCLAIMER}"
            logger.info("Added guardrail disclaimer to response")

        return {"reply": reply}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5204, reload=True)
