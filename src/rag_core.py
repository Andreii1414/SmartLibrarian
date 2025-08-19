from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Environment ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or environment.")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "book_summaries")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_oa = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Data models ----------
@dataclass(frozen=True)
class RetrievedDoc:
    title: str
    distance: float
    summary: str

@dataclass(frozen=True)
class Recommendation:
    reply_text: str
    chosen_title: Optional[str]
    full_summary: Optional[str]

# ---------- Chroma provider ----------
class ChromaProvider:
    """
    Creates and returns a Chroma persistent collection.
    """

    def __init__(self, persist_dir: Optional[str], collection_name: str, space: str = "cosine"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.space = space

    def client(self) -> chromadb.Client:
        """
        Create a Chroma client with the specified persistence directory.
        """
        settings = ChromaSettings(anonymized_telemetry=False)
        if self.persist_dir:
            return chromadb.PersistentClient(path=self.persist_dir, settings=settings)
        return chromadb.PersistentClient(settings=settings)

    def collection(self):
        """
        Get or create the Chroma collection with the specified name and metadata.
        """
        return self.client().get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.space}
        )

# ---------- Embeddings ----------
class Embedder:
    """
    OpenAI text embedder using the specified model.
    """

    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a sequence of texts using the OpenAI embeddings API.
        """
        if not texts:
            return []
        resp = _oa.embeddings.create(model=self.model, input=list(texts))
        data_sorted = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data_sorted]

    def embed_query(self, q: str) -> List[float]:
        return self.embed_texts([q])[0]

# ---------- Retriever ----------
class BookRetriever:
    """
    Retrieves book summaries from a Chroma collection using an embedder.
    """

    def __init__(self, provider: ChromaProvider, embedder: Embedder):
        self.provider = provider
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        """
        Retrieve top K book summaries based on the query using the embedder.
        """
        if not query.strip():
            return []
        qvec = self.embedder.embed_query(query)
        coll = self.provider.collection()
        res = coll.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        out: List[RetrievedDoc] = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return out
        for i in range(len(res["ids"][0])):
            title = res["metadatas"][0][i].get("title", "(untitled)")
            distance = float(res["distances"][0][i])
            summary = res["documents"][0][i]
            out.append(RetrievedDoc(title=title, distance=distance, summary=summary))
        return out

# ---------- Local summary store ----------
class SummaryStore:
    """
    Loads exact-title -> full summary map from a local JSON file (or empty fallback).
    """

    def __init__(self, json_path: str | Path = "book_summaries.json"):
        self.path = Path(json_path)

    def load_map(self) -> Dict[str, str]:
        """
        Load the summary map from the JSON file.
        """
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {x["title"]: x["summary"] for x in data if "title" in x and "summary" in x}
        return {}

    def get_by_title(self, title: str) -> str:
        """
        Get the full summary for an exact book title.
        """
        return self.load_map().get(title, "Summary not found for the exact title.")

# ---------- Recommender with tool-calling ----------
class Recommender:
    """
    Produces a conversational recommendation using OpenAI Chat and
    calls a local tool to fetch the full summary by exact title.
    """
    def __init__(self, chat_model: str = CHAT_MODEL, summary_store: Optional[SummaryStore] = None):
        self.chat_model = chat_model
        self.summary_store = summary_store or SummaryStore()

        self.tools = [{
            "type": "function",
            "function": {
                "name": "get_summary_by_title",
                "description": "Return the full summary for an exact book title from a local store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Exact book title to look up."}
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
        }]

    @staticmethod
    def _context_from_retrieved(docs: Sequence[RetrievedDoc], limit: int = 3) -> str:
        """
        Create a formatted context string from retrieved documents.
        """
        lines = []
        for d in list(docs)[:limit]:
            lines.append(f"- Title: {d.title}\n  Dist: {d.distance:.4f}\n  Summary: {d.summary}")
        return "\n".join(lines) if lines else "(no context)"

    def _system_prompt(self) -> str:
        """
        System prompt for the model to set the context and rules for book recommendations.
        """
        return (
            "You are BookBuddy, a helpful book-recommendation assistant. "
            "Speak concise, warm English. Recommend exactly ONE book first, then give a 2–3 sentence why, "
            "mention 2–3 key themes, and suggest one alternative only if helpful. "
            "After choosing a book, call the tool get_summary_by_title with the EXACT title you recommended."
        )

    def _user_prompt(self, user_query: str, retrieved_context: str) -> str:
        """
        User prompt that combines the user's query with retrieved context.
        """
        return (
            "User request: " + user_query + "\n\n"
            "Retrieved candidates (from a local vector store):\n" + retrieved_context + "\n\n"
            "Rules:\n"
            "- Prefer the closest thematic match (lower distance).\n"
            "- If multiple are close, pick the more accessible read.\n"
            "- Do not invent books; only use provided candidates.\n"
            "- Keep the recommendation under 120 words.\n"
            "- Then call the tool with the exact chosen title."
        )


    def _get_summary_by_title_local(self, title: str) -> str:
        """
        Fetch the full summary for an exact book title from the local summary store.
        :param title: Exact book title to look up.
        :return: Full summary string if found, otherwise a default message.
        """
        return self.summary_store.get_by_title(title)

    def recommend(self, user_query: str, retrieved: Sequence[RetrievedDoc]) -> Recommendation:
        """
        Generate a book recommendation based on the user's query and retrieved documents.
        """
        ctx = self._context_from_retrieved(retrieved, limit=3)
        system_msg = self._system_prompt()
        user_msg = self._user_prompt(user_query, ctx)

        first = _oa.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tools=self.tools,
            tool_choice="auto",
            temperature=0.7,
        )

        msg = first.choices[0].message
        assistant_text = (msg.content or "").strip()

        chosen_title = None
        full_summary = None

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "get_summary_by_title":
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    title = args.get("title")
                    if title:
                        chosen_title = title
                        full_summary = self._get_summary_by_title_local(title)
                    break

        return Recommendation(reply_text=assistant_text, chosen_title=chosen_title, full_summary=full_summary)

# ---------- Backward-compatible functions ----------
# Instantiate singletons once
_provider = ChromaProvider(persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME, space="cosine")
_embedder = Embedder(model=EMBED_MODEL)
_retriever = BookRetriever(provider=_provider, embedder=_embedder)

def retrieve_books(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """
    Backward-compatible function to retrieve book summaries.
    """
    docs = _retriever.retrieve(query, top_k=top_k)
    return [(d.title, d.distance, d.summary) for d in docs]

def generate_recommendation(
    user_query: str,
    retrieved: List[Tuple[str, float, str]],
    json_path: str | Path = "book_summaries.json",
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Backward-compatible function to generate a book recommendation.
    """
    docs = [RetrievedDoc(title=t, distance=dist, summary=s) for (t, dist, s) in retrieved]
    recommender = Recommender(chat_model=CHAT_MODEL, summary_store=SummaryStore(json_path))
    rec = recommender.recommend(user_query, docs)
    return rec.reply_text, rec.chosen_title, rec.full_summary
