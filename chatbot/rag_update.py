# chatbot/rag_update.py

import os
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings

# ── Load Environment ───────────────────────────────────────────────────────────
# .env is ../.env relative to this script
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
client = OpenAI(api_key=OPENAI_API_KEY)

# Where your docs live (20 pages max, as .md)
DOCS_DIR = Path(__file__).parent / "docs"
if not DOCS_DIR.exists():
    raise RuntimeError(f"Docs folder not found: {DOCS_DIR}")

# Where Chroma will persist its files (same as vector-search service)
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent / "chroma_data"))

# ── Initialize ChromaDB ────────────────────────────────────────────────────────
chroma_client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    )
)
# This will create (or open) the collection
collection = chroma_client.get_or_create_collection("deepmed_docs")

# ── Helper: Chunk text into ~1 000‑char pieces ──────────────────────────────────
def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    paras = text.split("\n\n")
    chunks, curr = [], ""
    for p in paras:
        if len(curr) + len(p) + 2 <= max_chars:
            curr += p + "\n\n"
        else:
            chunks.append(curr.strip())
            curr = p + "\n\n"
    if curr:
        chunks.append(curr.strip())
    return chunks

# ── Main: Read, embed, insert ──────────────────────────────────────────────────
def main():
    print(f"Resetting collection and writing into {PERSIST_DIR} …")
    chroma_client.reset()

    for md in sorted(DOCS_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text)):
            # 1) embed
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding = response.data[0].embedding

            # 2) insert
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": md.name, "chunk": idx}]
            )

    chroma_client.persist()
    print("✅ RAG database updated.")

if __name__ == "__main__":
    main()
