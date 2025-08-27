import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8000";

const defaultSettings = {
  top_k: 5,
  enable_voice: false,
  enable_tts: false,
  tts_voice: "alloy",
  tts_format: "mp3",
  enable_img: false,
  img_size: "1024x1024",
};

const ttsVoices = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"];
const ttsFormats = ["mp3", "wav", "aac", "flac", "opus", "pcm"];
const imgSizes = ["1024x1024", "1024x1792", "1792x1024"];

function Sidebar({ settings, setSettings }) {
  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">Settings</h2>
      <div className="sidebar-section">
        <label>
          <b>Top-K retrieved:</b> <span style={{ marginLeft: 2 }}>{settings.top_k}</span>
          <input
            type="range"
            min={1}
            max={10}
            value={settings.top_k}
            onChange={e => setSettings(s => ({ ...s, top_k: Number(e.target.value) }))}
            style={{ width: "100%" }}
          />
        </label>
        <div className="sidebar-hint">
          Retrieval ranks are based on cosine distance; lower is better.
        </div>
      </div>
      <div className="sidebar-section">
        <b>Text-to-Speech (TTS)</b>
        <div>
          <label className="styled-checkbox-label">
            <input
              type="checkbox"
              className="styled-checkbox"
              checked={settings.enable_tts}
              onChange={e => setSettings(s => ({ ...s, enable_tts: e.target.checked }))}
            />{" "}
            Enable Text-to-Speech
          </label>
        </div>
        <div>
          <label>
            Voice:{" "}
            <select
              className="styled-select"
              value={settings.tts_voice}
              onChange={e => setSettings(s => ({ ...s, tts_voice: e.target.value }))}
            >
              {ttsVoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        </div>
        <div>
          <label>
            Audio format:{" "}
            <select
              className="styled-select"
              value={settings.tts_format}
              onChange={e => setSettings(s => ({ ...s, tts_format: e.target.value }))}
            >
              {ttsFormats.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </label>
        </div>
        <div className="sidebar-hint">
          When enabled, the app will generate audio for the recommendation + full summary.
        </div>
      </div>
      <div className="sidebar-section">
        <b>Image Generation</b>
        <div>
          <label className="styled-checkbox-label">
            <input
              type="checkbox"
              className="styled-checkbox"
              checked={settings.enable_img}
              onChange={e => setSettings(s => ({ ...s, enable_img: e.target.checked }))}
            />{" "}
            Enable image generation
          </label>
        </div>
        <div>
          <label>
            Image size:{" "}
            <select
              className="styled-select"
              value={settings.img_size}
              onChange={e => setSettings(s => ({ ...s, img_size: e.target.value }))}
            >
              {imgSizes.map(sz => <option key={sz} value={sz}>{sz}</option>)}
            </select>
          </label>
        </div>
      </div>
    </aside>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  
  let mainContent = msg.content;
  let showExpander = false;
  if (
    msg.role === "assistant" &&
    msg.full_summary &&
    msg.chosen_title &&
    mainContent.includes(`**Full summary for _${msg.chosen_title}_:**`)
  ) {
   
    const marker = `**Full summary for _${msg.chosen_title}_:**`;
    const idx = mainContent.indexOf(marker);
    if (idx !== -1) {
      mainContent = mainContent.slice(0, idx).trim();
      showExpander = true;
    }
  }
  return (
    <div className={`chat-message ${isUser ? "user" : "assistant"}`}>
      <div className="chat-avatar">
        {isUser ? "🧑" : "🤖"}
      </div>
      <div className="chat-bubble">
        <div style={{ whiteSpace: "pre-line" }}>
          {mainContent}
        </div>
        {showExpander && (
          <details style={{ marginTop: 10 }}>
            <summary>Full summary for {msg.chosen_title}</summary>
            <div style={{ marginTop: 6, whiteSpace: "pre-line" }}>{msg.full_summary}</div>
          </details>
        )}
        {msg.retrieved && (
          <details style={{ marginTop: 10 }}>
            <summary>Show retrieved candidates</summary>
            {msg.retrieved.length === 0 && (
              <div>No candidates found. Try rephrasing, e.g., 'war and redemption', 'friendship and magic', 'dystopia and rebellion'.</div>
            )}
            {msg.retrieved.map(([title, dist, summary], i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <b>{i + 1}. {title}</b> — distance: <code>{dist.toFixed(4)}</code>
                <div style={{ fontSize: 13, color: "#bbb" }}>
                  {summary.length > 120 ? summary.slice(0, 120) + "…" : summary}
                </div>
              </div>
            ))}
          </details>
        )}
        {msg.audio_b64 && msg.audio_format && (
          <div style={{ marginTop: 10 }}>
            <audio controls src={`data:audio/${msg.audio_format};base64,${msg.audio_b64}`}></audio>

          </div>
        )}
        {msg.img_url && (
          <div style={{ marginTop: 10 }}>
            <img src={msg.img_url} alt="Book cover" style={{ maxWidth: "100%", borderRadius: 8 }} />
            {msg.img_caption && (
              <div style={{ fontSize: 13, color: "#bbb", marginTop: 2 }}>{msg.img_caption}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [settings, setSettings] = useState(defaultSettings);
  const [chat, setChat] = useState([]); // {role, content, ...extras}
  const [prompt, setPrompt] = useState("");
  // --- Microphone recording state ---
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);
  const chatBottomRef = useRef(null);

  React.useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chat]);

  // --- Microphone recording handlers ---
  const handleStartRecording = async () => {
    setTranscript("");
    setAudioChunks([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new window.MediaRecorder(stream);
      setMediaRecorder(recorder);
      recorder.start();
      setRecording(true);
      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = async () => {
        setRecording(false);
        setMediaRecorder(null);
        const audioBlob = new Blob(chunks, { type: "audio/webm" });
        setAudioChunks([]);
        await handleTranscribeBlob(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
    } catch (err) {
      alert("Microphone access denied or not available.");
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
    }
  };

  const handleTranscribeBlob = async (audioBlob) => {
    setLoading(true);
    setTranscript("");
    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");
      const res = await axios.post(`${API_URL}/transcribe`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setTranscript(res.data.transcript);
      setPrompt(res.data.transcript);
    } catch (err) {
      setTranscript("Transcription error: " + (err.response?.data?.error || err.message));
    }
    setLoading(false);
  };

  const handlePromptSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    const userMsg = { role: "user", content: prompt };
    setChat(prev => [...prev, userMsg]);
    setPrompt("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/recommend`, {
        prompt,
        top_k: settings.top_k,
        enable_tts: settings.enable_tts,
        tts_voice: settings.tts_voice,
        tts_format: settings.tts_format,
        enable_img: settings.enable_img,
        img_size: settings.img_size,
      });
      const data = res.data;
      const assistantMsg = {
        role: "assistant",
        content: data.reply,
        full_summary: data.full_summary,
        chosen_title: data.chosen_title,
        img_url: data.img_url,
        img_caption: data.img_caption,
        audio_b64: data.audio_b64,
        audio_format: data.audio_format,
        retrieved: data.retrieved,
      };
      setChat(prev => [...prev, assistantMsg]);
    } catch (err) {
      setChat(prev => [...prev, {
        role: "assistant",
        content: "Error: " + (err.response?.data?.error || err.message)
      }]);
    }
    setLoading(false);
  };

  return (
    <div className="app-root">
      <Sidebar settings={settings} setSettings={setSettings} />
      <main className="main-content">
        <div className="main-inner">
          <h1 className="main-title">📚 BookBuddy — AI Book Recommendations</h1>
          <div className="main-caption">
            Ask in natural English. Example: <i>I want a book about friendship and magic.</i>
          </div>
          <div>
            {chat.map((msg, i) => (
              <ChatMessage key={i} msg={msg} />
            ))}
            <div ref={chatBottomRef} />
          </div>
          <form onSubmit={handlePromptSubmit} className="chat-input-form">

              <button
                type="button"
                className={`styled-btn mic-btn`}
                onClick={recording ? handleStopRecording : handleStartRecording}
                disabled={loading}
                style={{
                  background: recording
                    ? "linear-gradient(90deg, #e53935 60%, #ff7043 100%)"
                    : "linear-gradient(90deg, #43a047 60%, #66bb6a 100%)"
                }}
                title={recording ? "Stop Recording" : "Start Recording"}
              >
                {recording ? (
                  <span role="img" aria-label="stop">⏹️</span>
                ) : (
                  <span role="img" aria-label="mic">🎙️</span>
                )}
              </button>

            <textarea
              rows={2}
              className="chat-input"
              placeholder="Tell me what you’re in the mood to read…"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !prompt.trim()}
              className="chat-send-btn styled-btn"
            >
              {loading ? "Loading..." : "Send"}
            </button>
          </form>


        </div>
      </main>
    </div>
  );
}

export default App;
