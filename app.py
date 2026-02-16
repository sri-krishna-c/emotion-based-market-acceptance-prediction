import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper

from voice_confidence import (
    extract_voice_features,
    detect_hesitation,
    analyze_text_certainty,
    estimate_confidence
)
from text_sentiment import classify_transcript_sentiment
from tone_analyzer import analyze_tone
from sarcasm_detector import detect_sarcasm


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Voice Emotional Intelligence",
    layout="centered"
)

st.title("🧠 Voice Emotional Intelligence")
st.caption("Deep Voice-Based Audience Understanding (Voice Only)")


# ===================== RECORD BUTTON =====================
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
    sf.write("input.wav", audio, SAMPLE_RATE)

    st.success("Recording complete")


    # ===================== TRANSCRIPTION =====================
    model = whisper.load_model("small")
    result = model.transcribe("input.wav", fp16=False)
    text = result["text"].strip()


    # ===================== ANALYSIS =====================
    voice = extract_voice_features("input.wav", text)
    hesitation = detect_hesitation(text)
    text_certainty = analyze_text_certainty(text)

    confidence = estimate_confidence(
        voice,
        hesitation,
        text_certainty
    )

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


    # ===================== FINAL DECISION =====================
    if sarcasm:
        final = "Negative"
        reason = sarcasm_reason

    elif neg > pos:
        final = "Negative"
        reason = "Negative wording detected in speech"

    elif sentiment == "Positive":
        if hesitation in ["Medium"] or confidence == "Low":
            final = "Neutral"
            reason = "Positive wording with hesitation or low commitment"
        else:
            final = "Positive"
            reason = "Positive wording supported by confident tone"

    else:
        final = "Neutral"
        reason = "Mixed or neutral audience response"


    # ===================== FINAL OUTPUT =====================
    st.subheader("🎯 Final Audience Response")

    if final == "Positive":
        st.success("🟢 Positive")
    elif final == "Negative":
        st.error("🔴 Negative")
    else:
        st.warning("🟡 Neutral")

    st.caption(reason)


    # ===================== ANALYZE DROPDOWN =====================
    with st.expander("🔍 Analyze"):
        st.markdown("### 🧾 Transcription")
        st.write(text)

        st.markdown("### 📊 Transcript Analysis")
        st.write({
            "Sentiment": sentiment,
            "Positive Words": pos,
            "Negative Words": neg,
            "Downgrade Words": down,
            "Contrast ('but') Detected": has_but
        })

        st.markdown("### 🎤 Voice Analysis")
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
