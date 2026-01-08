import streamlit as st
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import librosa
import numpy as np

st.set_page_config(page_title="Voice Emotion Demo", layout="centered")

st.title("🎤 Voice Emotion & Sentiment Detection")
st.write("This module contributes to **Emotion-Based Market Acceptance Prediction**")
st.write("Real-time voice analysis using Emotion + Arousal fusion")

# Load emotion model once
@st.cache_resource
def load_model():
    classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1  # CPU
    )
    return classifier

classifier = load_model()

st.subheader("🎙 Record Your Voice")

audio = mic_recorder(
    start_prompt="🎙 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=False
)

if audio is not None:
    audio_bytes = audio["bytes"]

    # Save audio file
    with open("input.wav", "wb") as f:
        f.write(audio_bytes)

    st.audio(audio_bytes, format="audio/wav")
    st.success("Audio recorded successfully!")

    # Load audio for energy calculation
    y, sr = librosa.load("input.wav", sr=None)
    energy = np.mean(librosa.feature.rms(y=y))

    with st.spinner("Analyzing emotion..."):
        result = classifier("input.wav")

    # Show raw top 3 predictions
    st.markdown("### 🔬 Raw Emotion Predictions (Top 3)")
    for r in result[:3]:
        st.write(f"**{r['label']}** → {r['score']:.2f}")

    top_result = max(result[:3], key=lambda x: x["score"])
    emotion = top_result["label"].lower()
    confidence = top_result["score"]

    # =======================
    # FINAL TUNED FUSION LOGIC
    # =======================

    # Strong Positive (High energy, any emotion)
    if energy > 0.035:
        sentiment = "Positive"

    # Moderate Positive (happy/surprised/calm with medium energy)
    elif emotion in ["happy", "surprised", "calm"] and energy > 0.02:
        sentiment = "Positive"

    # Strong Negative (sad/fearful/disgust with low energy)
    elif emotion in ["sad", "fearful", "disgust"] and energy < 0.02:
        sentiment = "Negative"

    # Low energy fallback
    elif energy < 0.015:
        sentiment = "Negative"

    # Neutral zone
    elif 0.015 <= energy <= 0.03:
        sentiment = "Neutral"

    else:
        sentiment = "Neutral"

    # =======================

    st.markdown("## 🔍 Final Prediction Result")
    st.write(f"**Detected Emotion:** `{emotion.upper()}`")
    st.write(f"**Emotion Confidence:** `{confidence:.2f}`")
    st.write(f"**Voice Energy Level (Arousal):** `{energy:.4f}`")
    st.write(f"**Overall Sentiment (Emotion + Arousal Fusion):** `{sentiment}`")

    if sentiment == "Positive":
        st.success("🟢 Positive market response expected")
    elif sentiment == "Negative":
        st.error("🔴 Negative market response expected")
    else:
        st.warning("🟡 Neutral market response")
