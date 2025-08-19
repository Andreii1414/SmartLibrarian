from __future__ import annotations
import os
import io
import wave
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# External services (already in your project)
from rag_core import retrieve_books, generate_recommendation
from moderation import moderate_text
from tts import synthesize_speech
from stt import transcribe_wav_bytes
from image_gen import generate_book_image

# Voice capture
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


# =========================
# Config & Boot
# =========================
class AppConfig:
    """
    Centralized configuration, mostly env + UI defaults.
    """
    def __init__(self) -> None:
        load_dotenv()
        self.book_json_path: str = os.getenv("BOOK_JSON_PATH", "book_summaries.json")
        self.tts_model: str = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
        self.default_img_size: str = "1024x1792"
        self.rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        self.voice_sample_rate: int = 16000
        self.voice_channels: int = 1

    @staticmethod
    def set_page() -> None:
        st.set_page_config(page_title="BookBuddy — RAG Chatbot", page_icon="📚")


# =========================
# Application State
# =========================
@dataclass
class AppState:
    """
    Session state wrapper with defaults + helpers.
    """
    messages: List[dict] = field(default_factory=list)
    voice_frames: List[bytes] = field(default_factory=list)
    voice_running: bool = False
    last_audio: Optional[bytes] = None
    last_results: List[Tuple[str, float, str]] = field(default_factory=list)
    voice_prompt: Optional[str] = None

    @staticmethod
    def load() -> "AppState":
        if "app_state" not in st.session_state:
            st.session_state["app_state"] = AppState()
        return st.session_state["app_state"]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})


# =========================
# Voice Utilities
# =========================
@dataclass
class AudioCollector:
    """
    Collect raw PCM16 mono samples from incoming frames.
    """
    frames: List[bytes]

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        Callback to process incoming audio frames.
        """
        pcm = frame.to_ndarray(format="s16")
        mono = pcm.mean(axis=0).astype("int16") if pcm.ndim == 2 else pcm.astype("int16")
        self.frames.append(mono.tobytes())
        return frame


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """
    Wrap raw PCM 16-bit mono into a WAV container (in-memory).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# =========================
# UI Widgets
# =========================
@dataclass
class SidebarControls:
    top_k: int
    enable_voice: bool
    enable_tts: bool
    tts_voice: str
    tts_format: str
    enable_img: bool
    img_size: str


def render_sidebar(cfg: AppConfig) -> SidebarControls:
    """
    Render the sidebar with configuration controls.
    """
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K retrieved", min_value=1, max_value=10, value=5, step=1)
        st.caption("Retrieval ranks are based on cosine distance; lower is better.")
        st.markdown("---")

        st.subheader("Voice mode (microphone)")
        enable_voice = st.checkbox("Enable Voice Mode", value=False, help="Speak to the chatbot using your mic.")
        st.caption("Click START to record, STOP to finish, then press 'Transcribe last recording'.")

        st.markdown("---")
        st.subheader("Text-to-Speech (TTS)")
        enable_tts = st.checkbox("Enable Text-to-Speech", value=False)
        tts_voice = st.selectbox("Voice", ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"], index=0)
        tts_format = st.selectbox("Audio format", ["mp3", "wav", "aac", "flac", "opus", "pcm"], index=0)
        st.caption("When enabled, the app will generate audio for the recommendation + full summary.")

        st.markdown("---")
        st.subheader("Image Generation")
        enable_img = st.checkbox("Enable image generation", value=False)
        img_size = st.selectbox("Image size", ["1024x1792", "1024x1024", "1792x1024"], index=0)

    return SidebarControls(
        top_k=top_k,
        enable_voice=enable_voice,
        enable_tts=enable_tts,
        tts_voice=tts_voice,
        tts_format=tts_format,
        enable_img=enable_img,
        img_size=img_size,
    )


def render_history(state: AppState) -> None:
    """
    Render the chat history from the application state.
    """
    for msg in state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_retrieved(results: List[Tuple[str, float, str]]) -> None:
    """
    Render the retrieved book candidates in an expandable section.
    """
    with st.expander("Show retrieved candidates"):
        if not results:
            st.write("No candidates found. Try rephrasing, e.g., 'war and redemption', 'friendship and magic', 'dystopia and rebellion'.")
        else:
            for rank, (title, dist, summary) in enumerate(results, start=1):
                st.markdown(f"**{rank}. {title}** — distance: `{dist:.4f}`")
                st.write(textwrap.shorten(summary.replace("\n", " "), width=260, placeholder="…"))


# =========================
# Main App
# =========================
def main() -> None:
    cfg = AppConfig()
    cfg.set_page()
    state = AppState.load()

    st.title("📚 BookBuddy — AI Book Recommendations")
    st.caption("Ask in natural English. Example: *I want a book about friendship and magic.*")

    controls = render_sidebar(cfg)
    render_history(state)

    if controls.enable_voice:
        st.info("Voice mode is ON. Use the built-in START/STOP buttons, then click **Transcribe last recording**.")

        collector = AudioCollector(frames=state.voice_frames)
        webrtc_streamer(
            key="bookbuddy-voice",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=cfg.rtc_cfg,
            media_stream_constraints={"audio": True, "video": False},
            audio_receiver_size=256,
            audio_frame_callback=collector.recv,
        )

        if st.button("Transcribe last recording"):
            raw = b"".join(state.voice_frames)
            state.voice_frames.clear()
            if not raw:
                st.warning("No audio captured. Please try recording again.")
            else:
                wav_bytes = pcm16_to_wav(raw, sample_rate=cfg.voice_sample_rate, channels=cfg.voice_channels)
                with st.spinner("Transcribing…"):
                    try:
                        transcript = transcribe_wav_bytes(wav_bytes)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        transcript = ""
                if transcript:
                    st.toast(f"You said: {transcript}", icon="🗣️")
                    state.voice_prompt = transcript
                    st.rerun()

    # ----- Chat input: prefer voice transcript if available -----
    voice_prompt = state.voice_prompt
    if voice_prompt:
        # Consume it exactly once
        state.voice_prompt = None
    typed_prompt = st.chat_input("Tell me what you’re in the mood to read…")
    prompt = voice_prompt or typed_prompt

    if prompt:
        # Show user message
        state.add_user(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Moderation (block if flagged)
        try:
            allowed, reason = moderate_text(prompt)
        except Exception as e:
            allowed, reason = False, f"Moderation service unavailable: {e}"

        if not allowed:
            reply = (
                "I’d be happy to help, but I can’t process that request. "
                "Please rephrase your message respectfully and without offensive language."
            )
            state.add_assistant(reply)
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.stop()

        # Retrieval
        with st.spinner("Searching the library…"):
            results = retrieve_books(prompt, top_k=controls.top_k)
        state.last_results = results
        render_retrieved(results)

        # Generation
        if results:
            with st.spinner("Asking BookBuddy…"):
                reply, chosen_title, full_summary = generate_recommendation(
                    prompt, results, json_path=cfg.book_json_path
                )
        else:
            reply, chosen_title, full_summary = (
                "I couldn't find a close match in the library. Could you try a simpler description of the themes you enjoy?",
                None,
                None,
            )

        # Compose final
        final_msg = reply
        if chosen_title and full_summary:
            final_msg += f"\n\n**Full summary for _{chosen_title}_:**\n{full_summary}"

        # Assistant reply
        state.add_assistant(final_msg)
        with st.chat_message("assistant"):
            st.markdown(final_msg)

            # Optional TTS
            audio_bytes = None
            if controls.enable_tts:
                with st.spinner("Generating audio…"):
                    try:
                        audio_bytes = synthesize_speech(
                            final_msg,
                            model=cfg.tts_model,
                            voice=controls.tts_voice,
                            fmt=controls.tts_format,
                        )
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
            if audio_bytes:
                state.last_audio = audio_bytes
                st.audio(audio_bytes, format=f"audio/{controls.tts_format}")
                st.download_button(
                    label="Download audio",
                    data=audio_bytes,
                    file_name=f"bookbuddy_recommendation.{controls.tts_format}",
                    mime=f"audio/{controls.tts_format}",
                    type="secondary",
                )

            # Optional Image Gen (requires title)
            if controls.enable_img and chosen_title:
                with st.spinner("Generating illustrative image…"):
                    try:
                        img_url = generate_book_image(chosen_title, size=controls.img_size)
                        st.image(img_url, caption=f"Illustration for '{chosen_title}'", use_container_width=True)
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # Rerender persisted panels on sidebar changes
    if not prompt and state.last_results:
        render_retrieved(state.last_results)
    if not prompt and state.last_audio and st.sidebar.checkbox("Show last audio player", value=False):
        st.audio(state.last_audio)


if __name__ == "__main__":
    main()
