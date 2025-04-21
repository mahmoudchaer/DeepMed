# chatbot/vector_search_service/init_db.py

import os
from pathlib import Path
import openai
from chromadb import Client
from chromadb.config import Settings

openai.api_key = os.getenv("OPENAI_API_KEY")
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_dir
))
collection = client.get_or_create_collection("deepmed_docs")

DOCS_DIR = Path(__file__).parent / "docs"

def chunk_text(text, max_len=1000):
    paras = text.split("\n\n")
    chunks, curr = [], ""
    for p in paras:
        if len(curr) + len(p) + 2 <= max_len:
            curr += p + "\n\n"
        else:
            chunks.append(curr.strip())
            curr = p + "\n\n"
    if curr:
        chunks.append(curr.strip())
    return chunks

def populate_db():
    print("Populating ChromaDB...")
    for md in DOCS_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            emb = openai.Embedding.create(
                model="text-embedding-3-small",
                input=[chunk]
            ).data[0].embedding

            collection.add(
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": md.name, "chunk": i}]
            )
    client.persist()
    print("✅ ChromaDB initialized.")

if __name__ == "__main__":
    populate_db()
