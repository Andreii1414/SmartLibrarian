from __future__ import annotations
import os
from typing import List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import base64
import tempfile

from rag_core import retrieve_books, generate_recommendation
from moderation import moderate_text
from tts import synthesize_speech
from image_gen import generate_book_image

# =========================
# Config & Boot
# =========================
class AppConfig:
    def __init__(self) -> None:
        load_dotenv()
        self.book_json_path: str = os.getenv("BOOK_JSON_PATH", "book_summaries.json")
        self.tts_model: str = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
        self.default_img_size: str = "1024x1024"
        self.voice_sample_rate: int = 16000
        self.voice_channels: int = 1
        self.asr_model: str = os.getenv("ASR_MODEL", "gpt-4o-transcribe")
        self.asr_language: Optional[str] = os.getenv("ASR_LANGUAGE")

cfg = AppConfig()

# =========================
# FastAPI Setup
# =========================
app = FastAPI(
    title="BookBuddy Backend",
    description="Backend API for BookBuddy React frontend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models for API
# =========================
class RecommendRequest(BaseModel):
    prompt: str
    top_k: int = 5
    enable_tts: bool = False
    tts_voice: Optional[str] = "alloy"
    tts_format: Optional[str] = "mp3"
    enable_img: bool = False
    img_size: Optional[str] = "1024x1024"

class RecommendResponse(BaseModel):
    reply: str
    chosen_title: Optional[str]
    full_summary: Optional[str]
    img_url: Optional[str] = None
    img_caption: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_format: Optional[str] = None
    retrieved: Optional[List[Tuple[str, float, str]]] = None

class TranscribeResponse(BaseModel):
    transcript: str

class SettingsResponse(BaseModel):
    tts_voices: List[str]
    tts_formats: List[str]
    img_sizes: List[str]

# =========================
# API Endpoints
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/settings", response_model=SettingsResponse)
def get_settings():
    return SettingsResponse(
        tts_voices=["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"],
        tts_formats=["mp3", "wav", "aac", "flac", "opus", "pcm"],
        img_sizes=["1024x1024", "1024x1792", "1792x1024"]
    )

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):

    # Moderation
    allowed, reason = moderate_text(req.prompt)
    if not allowed:
        return JSONResponse(
            status_code=400,
            content={"error": "Prompt flagged by moderation.", "reason": reason}
        )

    # Retrieval
    results = retrieve_books(req.prompt, top_k=req.top_k)

    # Generation
    if results:
        reply, chosen_title, full_summary = generate_recommendation(
            req.prompt, results, json_path=cfg.book_json_path
        )
    else:
        reply, chosen_title, full_summary = (
            "I couldn't find a close match in the library. Could you try a simpler description of the themes you enjoy?",
            None,
            None,
        )

    # Compose final message
    final_msg = reply
    if chosen_title and full_summary:
        final_msg += f"\n\n**Full summary for _{chosen_title}_:**\n{full_summary}"

    # TTS
    audio_b64 = None
    audio_format = req.tts_format
    if req.enable_tts:
        try:
            audio_bytes = synthesize_speech(
                final_msg,
                model=cfg.tts_model,
                voice=req.tts_voice,
                fmt=audio_format,
            )
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            audio_b64 = None

    # Image
    img_url = None
    img_caption = None
    if req.enable_img and chosen_title:
        try:
            img_url = generate_book_image(chosen_title, size=req.img_size)
            img_caption = f"Illustration for '{chosen_title}'"
        except Exception:
            img_url = None

    return RecommendResponse(
        reply=final_msg,
        chosen_title=chosen_title,
        full_summary=full_summary,
        img_url=img_url,
        img_caption=img_caption,
        audio_b64=audio_b64,
        audio_format=audio_format if audio_b64 else None,
        retrieved=results,
    )

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    # Write uploaded file to temp file
    suffix = Path(file.filename).suffix if file and file.filename else ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    from openai import OpenAI
    client = OpenAI()
    try:
        with open(tmp_path, "rb") as fh:
            transcript = client.audio.transcriptions.create(
                model=cfg.asr_model,
                file=fh,
                response_format="text",
                language=cfg.asr_language if cfg.asr_language else None,
            )
        transcript_text = transcript if isinstance(transcript, str) else str(transcript)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return TranscribeResponse(transcript=transcript_text)

if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)
