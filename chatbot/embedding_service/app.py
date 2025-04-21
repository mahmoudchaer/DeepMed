# chatbot/embedding_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@app.post("/embedding/generate", response_model=EmbeddingResponse)
async def generate_embedding(req: EmbeddingRequest):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text
        )
        return EmbeddingResponse(embedding=response.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5201, reload=True)
