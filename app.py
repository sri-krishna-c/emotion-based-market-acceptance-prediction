import os
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
import whisper

# =====================================================
# ENVIRONMENT DETECTION (CORRECT & RELIABLE)
# =====================================================
IS_CLOUD = not st.runtime.exists()  # True on Streamlit Cloud, False locally

# sounddevice works ONLY locally
if not IS_CLOUD:
    import sounddevice as sd

# =====================================================
# IMPORT CORE MODULES
# =====================================================
from voice_confidence import (
    extract_voice_features,
    detect_hesitation,
    analyze_text_certainty,
    estimate_confidence
)
from text_sentiment import classify_transcript_sentiment
from tone_analyzer import analyze_tone
from sarcasm_detector import detect_sarcasm

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Voice Emotional Intelligence",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Voice Emotional Intelligence")
st.caption("Audience & Market Acceptance Analysis (Local + Cloud Safe)")

# =====================================================
# LOAD WHISPER ONCE (CRITICAL FOR CLOUD)
# =====================================================
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")  # CPU-safe

model = load_whisper()

# =====================================================
# INPUT MODE
# =====================================================
st.subheader("🎛 Input Mode")

mode = st.radio(
    "Choose input method",
    ["Upload Audio (Cloud & Local)", "Live Record (Local only)"],
    disabled=IS_CLOUD
)

audio_path = None

# =====================================================
# UPLOAD MODE (CLOUD + LOCAL)
# =====================================================
if mode == "Upload Audio (Cloud & Local)":
    audio_file = st.file_uploader("📤 Upload WAV audio", type=["wav"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        st.success("Audio uploaded successfully")

# =====================================================
# LIVE RECORD MODE (LOCAL ONLY)
# =====================================================
if mode == "Live Record (Local only)" and not IS_CLOUD:
    if st.button("🎙 Record Voice (5 seconds)"):
        SAMPLE_RATE = 44100
        DURATION = 5

        st.info("Recording... Speak naturally")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1
        )
        sd.wait()

        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            audio_path = tmp.name

        st.success("Recording complete")

# =====================================================
# PROCESSING PIPELINE
# =====================================================
if audio_path:
    # ---------------- TRANSCRIPTION ----------------
    with st.spinner("📝 Transcribing speech..."):
        result = model.transcribe(audio_path, fp16=False)
        text = result["text"].strip()

    # ---------------- VOICE ANALYSIS ----------------
    voice = extract_voice_features(audio_path, text)
    hesitation = detect_hesitation(text)
    text_certainty = analyze_text_certainty(text)

    confidence = estimate_confidence(
        voice,
        hesitation,
        text_certainty
    )

    # ---------------- TEXT ANALYSIS ----------------
    sentiment, pos, neg, down, has_but = classify_transcript_sentiment(text)
    tone_info = analyze_tone(text, voice)

    sarcasm_info = detect_sarcasm(
        text,
        voice["energy_mean"],
        voice["pitch_std"],
        tone_info["fillers_detected"]
    )

    sarcasm = sarcasm_info["sarcasm"]
    sarcasm_reason = sarcasm_info["reason"]

    # =================================================
    # FINAL DECISION ENGINE
    # =================================================
    if sarcasm:
        final = "Negative"
        reason = sarcasm_reason
    elif neg > pos:
        final = "Negative"
        reason = "Negative wording detected in speech"
    elif sentiment == "Positive":
        if hesitation in ["Medium", "High"] or confidence == "Low":
            final = "Neutral"
            reason = "Positive wording with hesitation or low confidence"
        else:
            final = "Positive"
            reason = "Positive wording supported by confident delivery"
    else:
        final = "Neutral"
        reason = "Mixed or neutral audience response"

    # =================================================
    # FINAL OUTPUT
    # =================================================
    st.subheader("🎯 Final Audience Response")

    if final == "Positive":
        st.success("🟢 Positive")
    elif final == "Negative":
        st.error("🔴 Negative")
    else:
        st.warning("🟡 Neutral")

    st.caption(reason)

    # =================================================
    # DETAILED ANALYSIS
    # =================================================
    with st.expander("🔍 Detailed Analysis"):
        st.markdown("### 🧾 Transcription")
        st.write(text)

        st.markdown("### 📊 Text Sentiment")
        st.write({
            "Overall Sentiment": sentiment,
            "Positive Words": pos,
            "Negative Words": neg,
            "Downgrade Words": down,
            "Contrast ('but')": has_but
        })

        st.markdown("### 🎤 Voice Features")
        st.write({
            "Energy Mean": round(voice["energy_mean"], 4),
            "Pitch Std": round(voice["pitch_std"], 2),
            "Speaking Rate": round(voice["speaking_rate"], 2),
            "Hesitation Level": hesitation
        })

        st.markdown("### 🎭 Tone & Nuance")
        st.write({
            "Tone": tone_info["tone"],
            "Fillers Detected": tone_info["fillers_detected"],
            "Sarcasm Detected": sarcasm,
            "Sarcasm Reason": sarcasm_reason,
            "Confidence Level": confidence,
            "Text Certainty": text_certainty
        })

    # Cleanup temp file
    try:
        os.remove(audio_path)
    except:
        pass