# chatbot/chatbot_gateway/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import os
import sys
from pathlib import Path
import httpx
import logging
import random

# Add the parent directory to sys.path for keyvault import
sys.path.append(str(Path(__file__).parent.parent.parent))
import keyvault

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMB_URL       = "http://embedding_service:5201"
VEC_URL       = "http://vector_search_service:5202"
LLM_URL       = "http://llm_generator_service:5203"

# Use a default system prompt instead of trying to retrieve from Key Vault
# The secret name had invalid characters (hyphen not allowed in Key Vault secret names)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant for DeepMed. Only answer questions about the platform or medical AI. If off-topic, politely decline."
)

# Try to get the prompt from Key Vault using a valid secret name format if environment variable is not set
if not SYSTEM_PROMPT or SYSTEM_PROMPT == "You are a helpful assistant for DeepMed. Only answer questions about the platform or medical AI. If off-topic, politely decline.":
    try:
        kv_prompt = keyvault.getenv("SYSTEM_PROMPT_VALUE")
        if kv_prompt:
            SYSTEM_PROMPT = kv_prompt
            logger.info("Successfully loaded system prompt from Key Vault")
    except Exception as e:
        logger.warning(f"Failed to load system prompt from Key Vault: {e}. Using default.")

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

def apply_message_retention_strategy(messages: List[Message]) -> List[Message]:
    """
    Apply a balanced retention strategy to message history:
    - Always keep the system prompt
    - Keep the last 6 messages (100% retention)
    - Messages -7 to -12: 85% retention
    - Messages -13 to -20: 60% retention
    - Messages -21 to -30: 35% retention
    - Older than 30: 10-20% retention
    
    Returns a filtered list of messages according to the retention policy.
    """
    if not messages:
        return []
    
    # First separate system messages (which are always kept)
    system_messages = [msg for msg in messages if msg.role == "system"]
    non_system_messages = [msg for msg in messages if msg.role != "system"]
    total_non_system = len(non_system_messages)
    
    # Always keep the last 6 messages
    result = system_messages + ([] if total_non_system <= 6 else non_system_messages[-6:])
    
    # Apply retention probabilities to earlier messages
    if total_non_system > 6:
        # Messages -7 to -12: 85% retention
        start_idx = max(0, total_non_system - 12)
        end_idx = total_non_system - 6
        for i in range(start_idx, end_idx):
            if random.random() < 0.85:
                result.append(non_system_messages[i])
        
        # Messages -13 to -20: 60% retention
        start_idx = max(0, total_non_system - 20)
        end_idx = max(0, total_non_system - 12)
        for i in range(start_idx, end_idx):
            if random.random() < 0.60:
                result.append(non_system_messages[i])
        
        # Messages -21 to -30: 35% retention
        start_idx = max(0, total_non_system - 30)
        end_idx = max(0, total_non_system - 20)
        for i in range(start_idx, end_idx):
            if random.random() < 0.35:
                result.append(non_system_messages[i])
        
        # Older than 30: 10-20% retention (using 15%)
        start_idx = 0
        end_idx = max(0, total_non_system - 30)
        for i in range(start_idx, end_idx):
            if random.random() < 0.15:
                result.append(non_system_messages[i])
    
    # Sort messages to maintain chronological order
    # System messages stay at the beginning, then non-system messages in order
    final_result = system_messages + sorted(
        [msg for msg in result if msg.role != "system"],
        key=lambda x: messages.index(x)
    )
    
    logger.info(f"Message retention: {total_non_system} messages reduced to {len(final_result) - len(system_messages)} (plus {len(system_messages)} system messages)")
    return final_result

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    # Check dependencies if needed
    dependencies_status = {
        "embedding_service": "unknown",
        "vector_search_service": "unknown",
        "llm_generator_service": "unknown"
    }
    
    # Perform basic dependency checks
    async with httpx.AsyncClient(timeout=2.0) as client:
        for service, url in [
            ("embedding_service", f"{EMB_URL}/health"),
            ("vector_search_service", f"{VEC_URL}/health"),
            ("llm_generator_service", f"{LLM_URL}/health")
        ]:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    dependencies_status[service] = "healthy"
                else:
                    dependencies_status[service] = "unhealthy"
            except Exception:
                dependencies_status[service] = "unavailable"
    
    # Overall status is healthy if the gateway itself is running
    return {
        "status": "healthy",
        "service": "chatbot_gateway",
        "dependencies": dependencies_status
    }

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
        
        # Apply retention strategy to message history
        filtered_history = apply_message_retention_strategy(req.history)
        msgs += [m.dict() for m in filtered_history]
        
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

        return {"reply": reply}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5204, reload=True)
