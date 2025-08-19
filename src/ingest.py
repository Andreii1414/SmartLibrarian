from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from openai import OpenAI

# ----- Config -----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")  # optional
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "book_summaries")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or environment.")

# ----- Clients -----
oa = OpenAI(api_key=OPENAI_API_KEY)

def get_chroma_client() -> chromadb.Client:
    if CHROMA_DIR:
        return chromadb.PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
    return chromadb.PersistentClient(settings=ChromaSettings(anonymized_telemetry=False))

def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine pairs well with OpenAI embeddings
    )

# ----- Embeddings -----
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
    data_sorted = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data_sorted]

# ----- Ingest -----
def main():
    ap = argparse.ArgumentParser(description="Ingest book summaries into ChromaDB")
    ap.add_argument("--file", type=str, default="book_summaries.json", help="Path to JSON file")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data: List[Dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        if "title" not in item or "summary" not in item:
            raise ValueError(f"Invalid item at index {i}: expected keys 'title' and 'summary'.")

    texts = [f"Title: {b['title']}\nSummary: {b['summary']}" for b in data]
    vectors = embed_texts(texts)

    coll = get_collection()
    ids = [f"book-{i}" for i in range(len(data))]
    documents = [b["summary"] for b in data]
    metadatas = [{"title": b["title"]} for b in data]

    # Deterministic IDs -> re-running replaces existing entries
    coll.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(ids)} docs into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
