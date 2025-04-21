# chatbot/vector_search_service/init_db.py

import os
import uuid
from pathlib import Path

import chromadb
from openai import OpenAI

# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB persistent client
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("deepmed_docs")

DOCS_DIR = Path(__file__).parent / "docs"


def chunk_text(text: str, max_len: int = 1000) -> list[str]:
    paras = text.split("\n\n")
    chunks: list[str] = []
    curr = ""
    for p in paras:
        if len(curr) + len(p) + 2 <= max_len:
            curr += p + "\n\n"
        else:
            chunks.append(curr.strip())
            curr = p + "\n\n"
    if curr:
        chunks.append(curr.strip())
    return chunks


def populate_db() -> None:
    print("Populating ChromaDB...")
    for md in DOCS_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            # Use the OpenAI client to create an embedding
            response = client_openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            emb = response.data[0].embedding

            # Generate a unique ID and add to ChromaDB
            unique_id = str(uuid.uuid4())
            collection.add(
                ids=[unique_id],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": md.name, "chunk": i}]
            )

    client.persist()
    print("✅ ChromaDB initialized.")


if __name__ == "__main__":
    populate_db()
