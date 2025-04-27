# chatbot/vector_search_service/init_db.py

import os
import uuid
from pathlib import Path
import sys
import logging

import chromadb
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path for keyvault import
sys.path.append(str(Path(__file__).parent.parent.parent))
import keyvault

# ── Get API Key from Azure Key Vault ────────────────────────────────────────────
api_key = keyvault.getenv("OPENAI-API-KEY")
if not api_key:
    logger.error("Missing OPENAI-API-KEY in Key Vault")
    raise ValueError("OPENAI-API-KEY is required in Key Vault")

# ✅ Fix: Clean proxy environment variables to prevent OpenAI crash
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# ── Initialize OpenAI client ─────────────────────────────────────────────────────
client_openai = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized")

# ── Initialize ChromaDB Persistent Client ───────────────────────────────────────
persist_dir = os.getenv("CHROMAPERSISTDIR", "./chroma_data")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("deepmed_docs")
logger.info(f"ChromaDB initialized with persist directory: {persist_dir}")

# ── Set docs folder ──────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent / "docs"

# ── Helper: Chunk text into ~1,000-char pieces ───────────────────────────────────
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

# ── Populate ChromaDB ────────────────────────────────────────────────────────────
def populate_db() -> None:
    logger.info("Populating ChromaDB...")

    for md in DOCS_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        logger.info(f"Processing {md.name}")

        for i, chunk in enumerate(chunk_text(text)):
            # 1) Embed using OpenAI
            response = client_openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            emb = response.data[0].embedding

            # 2) Insert into ChromaDB
            unique_id = str(uuid.uuid4())
            collection.add(
                ids=[unique_id],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": md.name, "chunk": i}]
            )
            logger.info(f"Added chunk {i} from {md.name}")

    logger.info("✅ ChromaDB initialization completed.")

# ── Run script ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    populate_db()
