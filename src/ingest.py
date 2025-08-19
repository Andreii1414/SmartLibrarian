from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, RateLimitError, InternalServerError


# -------------------------------
# Models / Records
# -------------------------------

@dataclass(frozen=True)
class BookRecord:
    """
    Represents a book record with title and summary.
    """
    title: str
    summary: str


# -------------------------------
# IO / Loading
# -------------------------------

class JsonBookLoader:
    """Loads BookRecord objects from a JSON file."""

    @staticmethod
    def load(path: Path) -> List[BookRecord]:
        """
        Load book records from a JSON file.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        records: List[BookRecord] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "title" not in item or "summary" not in item:
                raise ValueError(f"Invalid item at index {i}: expected keys 'title' and 'summary'.")
            records.append(BookRecord(title=item["title"], summary=item["summary"]))
        return records


# -------------------------------
# Embeddings
# -------------------------------

class OpenAIEmbedder:
    """
    Thin wrapper for OpenAI embeddings with batching + simple retry.
    Default model: text-embedding-3-small.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 256):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = max(1, batch_size)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a list of texts using OpenAI embeddings.
        """
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            vectors.extend(self._embed_chunk(chunk))
        return vectors

    def _embed_chunk(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a single chunk of texts with retry logic for rate limits.
        """
        backoff = 1.0
        for attempt in range(5):
            try:
                resp = self.client.embeddings.create(model=self.model, input=list(texts))
                # Ensure order by index
                data_sorted = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in data_sorted]
            except (RateLimitError, APIStatusError, InternalServerError):
                time.sleep(backoff)
                backoff *= 2
        # Final attempt without catching to surface the error
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        data_sorted = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data_sorted]


# -------------------------------
# Vector Store (ChromaDB)
# -------------------------------

class ChromaRepository:
    """
    Repository for book summaries using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "book_summaries",
        distance: str = "cosine",
    ):
        settings = ChromaSettings(anonymized_telemetry=False)
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        else:
            self.client = chromadb.PersistentClient(settings=settings)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    def upsert(
        self,
        records: Sequence[BookRecord],
        embeddings: Sequence[Sequence[float]],
        deterministic_ids: bool = True,
    ) -> int:
        """
        Upsert book records and their embeddings into the ChromaDB collection.
        """
        if len(records) != len(embeddings):
            raise ValueError("Number of records and embeddings must match.")

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []

        for rec in records:
            rid = self._make_id(rec) if deterministic_ids else self._uuid_like(rec)
            ids.append(rid)
            docs.append(rec.summary)
            metas.append({"title": rec.title})

        self.collection.upsert(ids=ids, embeddings=list(embeddings), documents=docs, metadatas=metas)
        return len(ids)

    @staticmethod
    def _make_id(rec: BookRecord) -> str:
        """
        Stable ID based on title so re-ingest replaces the same doc.
        Uses sha1(title) to avoid special character issues.
        """
        h = hashlib.sha1(rec.title.strip().lower().encode("utf-8")).hexdigest()
        return f"book-{h}"

    @staticmethod
    def _uuid_like(rec: BookRecord) -> str:
        # Non-deterministic fallback if ever needed
        h = hashlib.sha1(f"{rec.title}|{time.time_ns()}".encode("utf-8")).hexdigest()
        return f"book-{h}"


# -------------------------------
# Ingest Pipeline
# -------------------------------

class IngestPipeline:
    """
    Coordinates: load -> embed -> upsert.
    """

    def __init__(self, embedder: OpenAIEmbedder, repo: ChromaRepository):
        self.embedder = embedder
        self.repo = repo

    @staticmethod
    def _to_embed_texts(records: Sequence[BookRecord]) -> List[str]:
        """
        Convert book records to texts for embedding.
        """
        return [f"Title: {r.title}\nSummary: {r.summary}" for r in records]

    def run(self, records: Sequence[BookRecord]) -> int:
        """
        Run the full ingestion pipeline: load, embed, and upsert records.
        """
        texts = self._to_embed_texts(records)
        print(f"Embedding {len(texts)} books with model '{self.embedder.model}' ...")
        vectors = self.embedder.embed(texts)
        print("Embeddings computed. Upserting into ChromaDB ...")
        n = self.repo.upsert(records, vectors, deterministic_ids=True)
        return n


# -------------------------------
# CLI
# -------------------------------

def main():
    """
    Main entry point for the ingestion script.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest book summaries into ChromaDB (class-based).")
    parser.add_argument("--file", type=str, default="book_summaries.json", help="Path to JSON file")
    parser.add_argument("--collection", type=str, default=os.getenv("CHROMA_COLLECTION_NAME", "book_summaries"))
    parser.add_argument("--persist-dir", type=str, default=os.getenv("CHROMA_PERSIST_DIRECTORY"))
    parser.add_argument("--embed-model", type=str, default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or environment.")

    path = Path(args.file)
    records = JsonBookLoader.load(path)

    embedder = OpenAIEmbedder(api_key=api_key, model=args.embed_model, batch_size=args.batch_size)
    repo = ChromaRepository(persist_dir=args.persist_dir, collection_name=args.collection, distance="cosine")

    pipeline = IngestPipeline(embedder, repo)
    n = pipeline.run(records)
    print(f"Done. Upserted {n} docs into collection '{args.collection}'.")


if __name__ == "__main__":
    main()

#python src/ingest.py --file data/book_summaries.json
