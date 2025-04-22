# chatbot/embedding_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
from pathlib import Path
from openai import OpenAI
import logging

# Add the parent directory to sys.path for keyvault import
sys.path.append(str(Path(__file__).parent.parent.parent))
import keyvault

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
client = OpenAI(api_key=keyvault.getenv("OPENAI-API-KEY"))

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@app.post("/embedding/generate", response_model=EmbeddingResponse)
async def generate_embedding(req: EmbeddingRequest):
    try:
        logger.info(f"Generating embedding for text: {req.text[:50]}...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text
        )
        embedding = response.data[0].embedding
        logger.info(f"Successfully generated embedding of dimension {len(embedding)}")
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "embedding"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting embedding service on port 5201")
    uvicorn.run("app:app", host="0.0.0.0", port=5201, reload=True)
