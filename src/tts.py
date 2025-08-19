from __future__ import annotations
import os
from typing import Literal

from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def synthesize_speech(
    text: str,
    model: str | None = None,
    voice: str | None = None,
    fmt: Literal["mp3", "wav"] = "mp3",
) -> bytes:
    """
    Convert text to speech and return audio bytes.
    """
    model = model or os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    voice = voice or os.getenv("TTS_VOICE", "alloy")
    fmt = fmt or os.getenv("TTS_FORMAT", "mp3")

    resp = _client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=fmt
    )
    return resp.read()
