from __future__ import annotations
import os
from typing import Dict, List, Tuple
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from openai import OpenAI

# Load env once
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "book_summaries")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or environment.")

# OpenAI clients
_oa = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Chroma helpers --------------------
def _chroma_client() -> chromadb.Client:
    if CHROMA_DIR:
        return chromadb.PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
    return chromadb.PersistentClient(settings=ChromaSettings(anonymized_telemetry=False))


def _collection():
    return _chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ---------- Embeddings ----------

def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _oa.embeddings.create(model=EMBED_MODEL, input=texts)
    data_sorted = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data_sorted]

def embed_query(q: str) -> List[float]:
    return _embed_texts([q])[0]


# ---------- Retrieval ----------

def retrieve_books(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """
    Semantic search by theme/context.
    Returns (title, distance, summary). Lower distance -> closer match (cosine).
    """
    if not query.strip():
        return []
    qvec = embed_query(query)
    coll = _collection()
    res = coll.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )
    out: List[Tuple[str, float, str]] = []
    if not res or not res.get("ids") or not res["ids"][0]:
        return out
    for i in range(len(res["ids"][0])):
        title = res["metadatas"][0][i].get("title", "(untitled)")
        distance = float(res["distances"][0][i])
        summary = res["documents"][0][i]
        out.append((title, distance, summary))
    return out


# -------------------- Local summary store --------------------
def _load_summaries_map(json_path: Path) -> Dict[str, str]:
    """
    Load title -> full summary map from a local JSON file.
    If file not found or invalid, fall back to an empty map.
    """
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Expect [{"title": "...", "summary": "..."}]
        if isinstance(data, list):
            return {item["title"]: item["summary"] for item in data if "title" in item and "summary" in item}
    return {}

def get_summary_by_title(title: str, json_path: Path) -> str:
    """
    Return the full summary for an exact title from the local store.
    """
    store = _load_summaries_map(json_path)
    return store.get(title, "Summary not found for the exact title.")

# Tool schema for OpenAI function calling
GET_SUMMARY_TOOL = [{
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
            "additionalProperties": False
        }
    }
}]

# -------------------- Generation with tool-calling --------------------
def generate_recommendation(
    user_query: str,
    retrieved: List[Tuple[str, float, str]],
    json_path: str | Path = "book_summaries.json",
) -> Tuple[str, str | None, str | None]:
    """
    Ask the model to recommend ONE title from retrieved candidates and call get_summary_by_title(title).
    Returns: (assistant_reply, chosen_title, full_summary)
    """
    json_path = Path(json_path)
    # Prepare a compact context for the model
    context_lines = []
    for (title, dist, summary) in retrieved[:3]:
        context_lines.append(f"- Title: {title}\n  Dist: {dist:.4f}\n  Summary: {summary}")
    context = "\n".join(context_lines) if context_lines else "(no context)"

    system_msg = (
        "You are BookBuddy, a helpful book-recommendation assistant. "
        "Speak concise, warm English. Recommend exactly ONE book first, then give a 2–3 sentence why, "
        "mention 2–3 key themes, and suggest one alternative only if helpful. "
        "After choosing a book, call the tool get_summary_by_title with the EXACT title you recommended."
    )
    user_msg = (
        "User request: " + user_query + "\n\n"
        "Retrieved candidates (from a local vector store):\n" + context + "\n\n"
        "Rules:\n"
        "- Prefer the closest thematic match (lower distance).\n"
        "- If multiple are close, pick the more accessible read.\n"
        "- Do not invent books; only use provided candidates.\n"
        "- Keep the recommendation under 120 words.\n"
        "- Then call the tool with the exact chosen title."
    )

    # First model call with tools enabled
    first = _oa.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        tools=GET_SUMMARY_TOOL,
        tool_choice="auto",
        temperature=0.7,
    )

    assistant_msg = first.choices[0].message
    assistant_text = (assistant_msg.content or "").strip()

    chosen_title = None
    full_summary = None

    if assistant_msg.tool_calls:
        # Expect exactly one tool call with {"title": "..."}
        for tc in assistant_msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                title = args.get("title")
                if title:
                    chosen_title = title
                    # Execute the local tool
                    full_summary = get_summary_by_title(title, json_path)
                break

    return assistant_text, chosen_title, full_summary

#python src/ingest.py --file data/book_summaries.json