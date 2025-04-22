# chatbot/vector_search_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
import chromadb
import logging
from pathlib import Path

# Add the parent directory to sys.path for keyvault import
sys.path.append(str(Path(__file__).parent.parent.parent))
import keyvault

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
persist_dir = keyvault.getenv("CHROMAPERSISTDIR", "./chroma_data")
logger.info(f"Using ChromaDB persist directory: {persist_dir}")

try:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("deepmed_docs")
    logger.info(f"Successfully connected to ChromaDB collection")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise


class SearchRequest(BaseModel):
    embedding: List[float]
    k: int = 4


class SearchResponse(BaseModel):
    documents: List[str]
    distances: List[float]


@app.post("/vector/search", response_model=SearchResponse)
async def vector_search(req: SearchRequest):
    try:
        logger.info(f"Searching for similar documents with k={req.k}")
        if len(req.embedding) < 1:
            raise ValueError("Empty embedding provided")
            
        res = collection.query(
            query_embeddings=[req.embedding],
            n_results=req.k
        )
        
        logger.info(f"Found {len(res['documents'][0])} documents")
        return SearchResponse(
            documents=res["documents"][0],
            distances=res["distances"][0]
        )
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        # Simple count to verify we can query the collection
        count = collection.count()
        return {
            "status": "healthy", 
            "service": "vector_search",
            "document_count": count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting vector search service on port 5202")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5202,
        reload=True,
        log_level="info"
    )
