from __future__ import annotations
import os
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv


from rag_core import retrieve_books, generate_recommendation
from moderation import moderate_text
from tts import synthesize_speech
from image_gen import generate_book_image

import tempfile
from openai import OpenAI

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
        self.default_img_size: str = "1024x1024"
        self.voice_sample_rate: int = 16000
        self.voice_channels: int = 1

        self.asr_model: str = os.getenv("ASR_MODEL", "gpt-4o-transcribe")
        self.asr_language: Optional[str] = os.getenv("ASR_LANGUAGE")

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

    def add_assistant(
        self,
        content: str,
        audio_bytes: Optional[bytes] = None,
        audio_format: Optional[str] = None,
        img_url: Optional[str] = None,
        img_caption: Optional[str] = None,
        retrieved: Optional[List[Tuple[str, float, str]]] = None,
    ) -> None:
        self.messages.append({
            "role": "assistant",
            "content": content,
            "audio_bytes": audio_bytes,
            "audio_format": audio_format,
            "img_url": img_url,
            "img_caption": img_caption,
            "retrieved": retrieved,
        })


# =========================
# Voice Utilities
# =========================
def transcribe_upload(file, cfg: AppConfig) -> str:
    """
    Transcribe an uploaded audio/video file using OpenAI's transcription API.
    Accepts common formats (mp3, mp4, wav, m4a, webm, ogg, flac, aac, opus).
    Returns plain text.
    """
    client = OpenAI()

    # Write the uploaded bytes to a temp file so the SDK can read it
    suffix = Path(file.name).suffix if file and file.name else ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as fh:
            transcript = client.audio.transcriptions.create(
                model=cfg.asr_model,
                file=fh,
                # Optional settings:
                response_format="text",
                language=cfg.asr_language if cfg.asr_language else None,
            )
        # The SDK returns a string when response_format="text"
        return transcript if isinstance(transcript, str) else str(transcript)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

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

        st.markdown("---")
        st.subheader("Text-to-Speech (TTS)")
        enable_tts = st.checkbox("Enable Text-to-Speech", value=False)
        tts_voice = st.selectbox("Voice", ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"], index=0)
        tts_format = st.selectbox("Audio format", ["mp3", "wav", "aac", "flac", "opus", "pcm"], index=0)
        st.caption("When enabled, the app will generate audio for the recommendation + full summary.")

        st.markdown("---")
        st.subheader("Image Generation")
        enable_img = st.checkbox("Enable image generation", value=False)
        img_size = st.selectbox("Image size", ["1024x1024", "1024x1792", "1792x1024"], index=0)

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
    Render the chat history from the application state, including audio, images, and retrieved candidates.
    """
    for msg in state.messages:
        with st.chat_message(msg["role"]):
            # Split out full summary if present
            content = msg["content"]
            summary_marker = "\n\n**Full summary for _"
            if summary_marker in content:
                main_part, summary_part = content.split(summary_marker, 1)
                st.markdown(main_part)
                summary_title_end = summary_part.find("_:**\n")
                if summary_title_end != -1:
                    summary_title = summary_part[:summary_title_end]
                    summary_text = summary_part[summary_title_end + 5 :]
                    with st.expander(f"Full summary for {summary_title.strip()}"):
                        st.markdown(summary_text)
                else:
                    st.markdown(summary_marker + summary_part)
            else:
                st.markdown(content)
            if msg.get("retrieved"):
                with st.expander("Show retrieved candidates"):
                    results = msg["retrieved"]
                    if not results:
                        st.write("No candidates found. Try rephrasing, e.g., 'war and redemption', 'friendship and magic', 'dystopia and rebellion'.")
                    else:
                        for rank, (title, dist, summary) in enumerate(results, start=1):
                            st.markdown(f"**{rank}. {title}** — distance: `{dist:.4f}`")
                            st.write(textwrap.shorten(summary.replace("\n", " "), width=260, placeholder="…"))
            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format=f"audio/{msg.get('audio_format','mp3')}")
                st.download_button(
                    label="Download audio",
                    data=msg["audio_bytes"],
                    file_name=f"bookbuddy_recommendation.{msg.get('audio_format','mp3')}",
                    mime=f"audio/{msg.get('audio_format','mp3')}",
                    type="secondary",
                )
            if msg.get("img_url"):
                st.image(msg["img_url"], caption=msg.get("img_caption", ""), use_container_width=True)


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

    # ----- Voice mode -----
    if controls.enable_voice:
        st.subheader("🎤 Upload audio/video for transcription")
        uploaded = st.file_uploader(
            "Drop an audio/video file (mp3, mp4, wav, m4a, webm, ogg, flac, aac, opus)",
            type=["mp3", "mp4", "wav", "m4a", "webm", "ogg", "flac", "aac", "opus"],
            accept_multiple_files=False,
            help="After uploading, click Transcribe to convert speech to text and use it as your chat prompt."
        )

        if uploaded:
            col1, col2 = st.columns([1, 3])
            with col1:
                transcribe_clicked = st.button("Transcribe", type="primary")
            with col2:
                st.caption(f"Selected file: **{uploaded.name}**")

            if transcribe_clicked:
                with st.spinner("Transcribing…"):
                    try:
                        transcript_text = transcribe_upload(uploaded, cfg)
                        state.voice_prompt = transcript_text.strip()
                        with st.expander("Show transcript"):
                            st.write(transcript_text)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

    # ----- Chat input -----
    voice_prompt = state.voice_prompt
    if voice_prompt:
        state.voice_prompt = None
    typed_prompt = st.chat_input("Tell me what you’re in the mood to read…")
    prompt = voice_prompt or typed_prompt

    # Show chat history up to now (so user message appears immediately)
    render_history(state)

    if prompt:
        # Show user message immediately
        state.add_user(prompt)
        st.rerun()

    # If the last message is from the user, generate assistant reply
    if state.messages and state.messages[-1]["role"] == "user":
        prompt = state.messages[-1]["content"]

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
            st.rerun()

        # Retrieval
        with st.spinner("Searching the library…"):
            results = retrieve_books(prompt, top_k=controls.top_k)

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

        # Prepare extras
        audio_bytes = None
        audio_format = controls.tts_format
        if controls.enable_tts:
            with st.spinner("Generating audio…"):
                try:
                    audio_bytes = synthesize_speech(
                        final_msg,
                        model=cfg.tts_model,
                        voice=controls.tts_voice,
                        fmt=audio_format,
                    )
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                    audio_bytes = None

        img_url = None
        img_caption = None
        if controls.enable_img and chosen_title:
            with st.spinner("Generating illustrative image…"):
                try:
                    img_url = generate_book_image(chosen_title, size=controls.img_size)
                    img_caption = f"Illustration for '{chosen_title}'"
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    img_url = None

        # Store everything in state
        state.add_assistant(
            final_msg,
            audio_bytes=audio_bytes,
            audio_format=audio_format if audio_bytes else None,
            img_url=img_url,
            img_caption=img_caption,
            retrieved=results,
        )
        st.rerun()


if __name__ == "__main__":
    main()
