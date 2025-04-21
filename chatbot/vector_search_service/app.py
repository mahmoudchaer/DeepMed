# chatbot/vector_search_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from chromadb import Client
from chromadb.config import Settings

app = FastAPI()
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=os.getenv("CHROMA_PERSIST_DIR","./chroma_data")
))
collection = client.get_or_create_collection("deepmed_docs")

class SearchRequest(BaseModel):
    embedding: list[float]
    k: int = 4

class SearchResponse(BaseModel):
    documents: list[str]
    distances: list[float]

@app.post("/vector/search", response_model=SearchResponse)
async def vector_search(req: SearchRequest):
    try:
        res = collection.query(
            query_embeddings=[req.embedding],
            n_results=req.k
        )
        return SearchResponse(
            documents=res["documents"][0],
            distances=res["distances"][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5202, reload=True)
