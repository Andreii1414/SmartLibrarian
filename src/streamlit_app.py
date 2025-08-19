from __future__ import annotations
import os
from pathlib import Path
import textwrap
import wave
import io
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from rag_core import retrieve_books, generate_recommendation
from moderation import moderate_text
from tts import synthesize_speech

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

from stt import transcribe_wav_bytes

from image_gen import generate_book_image

load_dotenv()

st.set_page_config(page_title="BookBuddy — RAG Chatbot", page_icon="📚")

if "voice_frames" not in st.session_state:
    st.session_state.voice_frames = []
if "voice_running" not in st.session_state:
    st.session_state.voice_running = False
if "voice_sample_rate" not in st.session_state:
    st.session_state.voice_sample_rate = 16000
if "voice_channels" not in st.session_state:
    st.session_state.voice_channels = 1

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieved", min_value=1, max_value=10, value=5, step=1)
    st.caption("Retrieval ranks are based on cosine distance; lower is better.")
    st.markdown("---")

    st.subheader("Voice mode (microphone)")
    enable_voice = st.checkbox("Enable Voice Mode", value=False, help="Speak to the chatbot using your mic.")
    st.caption("Press 'Start' below, speak, then press 'Stop & Transcribe' to send your question.")

    st.markdown("---")
    st.subheader("Text-to-Speech (TTS)")
    enable_tts = st.checkbox("Enable Text-to-Speech", value=False)
    tts_voice = st.selectbox("Voice", ["alloy", "verse", "aria", "coral", "sage"], index=0)
    tts_format = st.selectbox("Audio format", ["mp3", "wav"], index=0)
    st.caption("When enabled, the app will generate audio for the recommendation + full summary.")

    st.markdown("---")
    st.subheader("Image Generation")
    enable_img = st.checkbox("Enable image generation", value=False)
    img_size = st.selectbox(
        "Image size",
        ["1024x1024", "1024x1792", "1792x1024"],
        index=0
    )

BOOK_JSON_PATH = os.getenv("BOOK_JSON_PATH", "book_summaries.json")

st.title("📚 BookBuddy — AI Book Recommendations")
st.caption("Ask in natural English. Example: *I want a book about friendship and magic.*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """
    Wrap raw PCM 16-bit mono samples into a WAV container in-memory.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

if enable_voice:
    st.info("Voice mode is ON. Click **Start** below, speak, then click **Transcribe last recording**.")
    # Minimal TURN/STUN config: using Google STUN (public). For production, use your own TURN.
    RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    @dataclass
    class AudioCollector:
        """
        Collects raw PCM samples from incoming audio frames (mono, 16kHz).
        """
        frames: list

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray(format="s16")
            if pcm.ndim == 2:
                mono = pcm.mean(axis=0).astype("int16")
            else:
                mono = pcm.astype("int16")
            self.frames.append(mono.tobytes())
            return frame

    # The processor that streamlit-webrtc will call for each audio frame
    collector = AudioCollector(frames=st.session_state.voice_frames)

    webrtc_ctx = webrtc_streamer(
        key="bookbuddy-voice",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CFG,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=256,
        audio_frame_callback=collector.recv,
    )

    if st.button("Transcribe last recording"):
        raw = b"".join(st.session_state.voice_frames)
        st.session_state.voice_frames.clear()
        if not raw:
            st.warning("No audio captured. Please try recording again.")
        else:
            wav_bytes = _pcm16_to_wav(raw, sample_rate=16000, channels=1)
            with st.spinner("Transcribing…"):
                try:
                    transcribed = transcribe_wav_bytes(wav_bytes)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    transcribed = ""
            if transcribed:
                st.toast(f"You said: {transcribed}", icon="🗣️")
                st.session_state["voice_prompt"] = transcribed
                st.rerun()



# Chat input
prompt = st.chat_input("Tell me what you’re in the mood to read…")
if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        allowed, reason = moderate_text(prompt)
    except Exception as e:
        allowed, reason = False, f"Moderation service unavailable: {e}"

    if not allowed:
        reply = (
            "I’d be happy to help, but I can’t process that request. "
            "Please rephrase your message respectfully and without offensive language."
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.stop()

    # Retrieval step
    with st.spinner("Searching the library…"):
        results = retrieve_books(prompt, top_k=top_k)

    # Show retrieved context in an expander
    with st.expander("Show retrieved candidates"):
        if not results:
            st.write("No candidates found. Try rephrasing, e.g., 'war and redemption', 'friendship and magic', 'dystopia and rebellion'.")
        else:
            for rank, (title, dist, summary) in enumerate(results, start=1):
                st.markdown(f"**{rank}. {title}** — distance: `{dist:.4f}`")
                st.write(textwrap.shorten(summary.replace("\n", " "), width=260, placeholder="…"))

    # Generation step
    if results:
        with st.spinner("Asking BookBuddy…"):
            reply, chosen_title, full_summary = generate_recommendation(
                prompt, results, json_path=BOOK_JSON_PATH
            )
    else:
        reply, chosen_title, full_summary = (
            "I couldn't find a close match in the library. Could you try a simpler description of the themes you enjoy?",
            None,
            None,
        )

    # Append assistant message
    final_msg = reply
    if chosen_title and full_summary:
        final_msg += f"\n\n**Full summary for _{chosen_title}_:**\n{full_summary}"

    st.session_state.messages.append({"role": "assistant", "content": final_msg})
    with st.chat_message("assistant"):
        st.markdown(final_msg)

        if enable_tts:
            audio_text = final_msg

            with st.spinner("Generating audio…"):
                try:
                    audio_bytes = synthesize_speech(
                        audio_text,
                        model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
                        voice=tts_voice,
                        fmt=tts_format,
                    )
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                    audio_bytes = None

            if audio_bytes:
                st.audio(audio_bytes, format=f"audio/{tts_format}")

                filename = f"bookbuddy_recommendation.{tts_format}"
                st.download_button(
                    label="Download audio",
                    data=audio_bytes,
                    file_name=filename,
                    mime=f"audio/{tts_format}",
                    type="secondary",
                )
        if enable_img and chosen_title and full_summary:
            with st.spinner("Generating illustrative image…"):
                try:
                    img_url = generate_book_image(chosen_title, full_summary, size=img_size)
                    st.image(img_url, caption=f"Illustration for '{chosen_title}'", use_container_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")