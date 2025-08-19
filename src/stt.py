from __future__ import annotations
import io
import os
from typing import Optional
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def transcribe_wav_bytes(
    wav_bytes: bytes,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe WAV audio bytes and return the recognized text.
    """
    model = model or os.getenv("STT_MODEL", "whisper-1")
    language = language or os.getenv("STT_LANGUAGE")

    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"  # name hint for the API

    resp = _client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        **({"language": language} if language else {}),
    )
    return resp.text.strip()
