# chatbot/rag_update.py

import os
import sys
import logging
import httpx
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path for keyvault import
sys.path.append(str(Path(__file__).parent.parent))
import keyvault

from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings

# Get API key from Key Vault
api_key = keyvault.getenv("OPENAI-API-KEY")
if not api_key:
    logger.error("Missing OPENAI-API-KEY in Key Vault")
    raise RuntimeError("Missing OPENAI-API-KEY in Key Vault")

# Clean proxy environment variables to prevent OpenAI crash
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# Initialize OpenAI client with custom httpx client to avoid proxy issues
http_client = httpx.Client()
client = OpenAI(
    api_key=api_key,
    http_client=http_client
)
logger.info("OpenAI client initialized")

# Where your docs live (20 pages max, as .md)
DOCS_DIR = Path(__file__).parent / "vector_search_service" / "docs"
if not DOCS_DIR.exists():
    raise RuntimeError(f"Docs folder not found: {DOCS_DIR}")

# Where Chroma will persist its files (same as vector-search service)
PERSIST_DIR = str(Path(__file__).parent / "vector_search_service" / "chroma_data")
logger.info(f"Using ChromaDB persist directory: {PERSIST_DIR}")

# ── Initialize ChromaDB ────────────────────────────────────────────────────────
chroma_client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    )
)
collection = chroma_client.get_or_create_collection("deepmed_docs")
logger.info("ChromaDB collection initialized")

# ── Helper: Chunk text into ~1,000‑char pieces ─────────────────────────────────
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
    logger.info(f"Resetting collection and writing into {PERSIST_DIR} …")
    chroma_client.reset()

    for md in sorted(DOCS_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        logger.info(f"Processing {md.name}")
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
            logger.info(f"Added chunk {idx} from {md.name}")

    chroma_client.persist()
    logger.info("✅ RAG database updated.")

if __name__ == "__main__":
    main()
