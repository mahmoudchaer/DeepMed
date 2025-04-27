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

# Improve API key retrieval with better error handling
api_key = keyvault.getenv("OPENAI-API-KEY")
if not api_key:
    logger.error("Failed to retrieve OPENAI-API-KEY from Key Vault")
    # Try environment variable as fallback (check both formats)
    api_key = os.getenv("OPENAI-API-KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Using OPENAI API key from environment variable")
    else:
        logger.error("OPENAI API key not found in Key Vault or environment variables")
        raise ValueError("OPENAI API key is required")
else:
    # Log the first few characters of the API key to verify format (safely)
    key_prefix = api_key[:7] + "..." if len(api_key) > 10 else "invalid_key"
    logger.info(f"Retrieved API key from Key Vault with prefix: {key_prefix}")

# Initialize the OpenAI client
try:
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

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

@app.get("/diagnostics")
async def diagnostics():
    # Create a safe version of the API key for diagnostics
    key_info = {
        "key_source": "keyvault" if keyvault.getenv("OPENAI-API-KEY") else "environment",
        "key_format": "project" if api_key and api_key.startswith("sk-proj-") else "standard",
        "key_prefix": api_key[:7] + "..." if api_key and len(api_key) > 10 else "invalid_key",
        "key_length": len(api_key) if api_key else 0,
        "env_vars_set": {
            "OPENAI-API-KEY": bool(os.getenv("OPENAI-API-KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY"))
        }
    }
    
    # Test the API key with a simple request
    test_result = "not_tested"
    try:
        # Make a minimal API call to test authentication
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        test_result = "success"
    except Exception as e:
        test_result = f"error: {str(e)}"
    
    return {
        "api_key_info": key_info,
        "api_test_result": test_result,
        "client_type": str(type(client)),
        "openai_module_version": OpenAI.__version__
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting embedding service on port 5201")
    uvicorn.run("app:app", host="0.0.0.0", port=5201, reload=True)
