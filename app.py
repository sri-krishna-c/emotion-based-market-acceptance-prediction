import streamlit as st
import whisper
from transformers import pipeline
import librosa
import numpy as np
from deepface import DeepFace
import cv2
from capture_module import capture_both

st.set_page_config(page_title="Multimodal Emotional Intelligence", layout="centered")

# ================== STYLE ==================
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.section-card {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='title'>🧠 Multimodal Emotional Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Voice + Face Emotion Fusion for Market Acceptance Prediction</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    voice_emotion_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return whisper_model, voice_emotion_model, sentiment_model

whisper_model, voice_emotion_model, sentiment_model = load_models()

# ================== SESSION STATE ==================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ================== CAPTURE ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🎥 Live Face + 🎙 Voice Capture")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start 5s Face + Voice Capture"):
        with st.spinner("Recording face and voice..."):
            audio_path, video_path = capture_both()
            st.session_state.audio_path = audio_path
            st.session_state.video_path = video_path
        st.success("Capture complete")

with col2:
    if st.button("🔁 Retry Capture"):
        st.session_state.audio_path = None
        st.session_state.video_path = None
        st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ================== PLAYBACK + PROCESS ==================
if st.session_state.audio_path and st.session_state.video_path:

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("▶ Playback")
    st.video(st.session_state.video_path)
    st.audio(st.session_state.audio_path)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================== AUDIO ENERGY ==================
    y, sr = librosa.load(st.session_state.audio_path, sr=None)
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # ================== VOICE EMOTION ==================
    voice_result = voice_emotion_model(st.session_state.audio_path)
    voice_top = max(voice_result, key=lambda x: x["score"])
    voice_emotion = voice_top["label"].lower()
    voice_conf = voice_top["score"]

    if voice_conf < 0.3:
        voice_emotion = "uncertain"

    # ================== SPEECH TO TEXT ==================
    transcription = whisper_model.transcribe(st.session_state.audio_path)
    text_output = transcription["text"]
    text_lower = text_output.lower()

    # ================== TEXT SENTIMENT ==================
    text_sentiment = sentiment_model(text_output)[0]
    text_label = text_sentiment["label"]
    text_conf = text_sentiment["score"]

    # ================== FACE EMOTION ==================
    cap = cv2.VideoCapture(st.session_state.video_path)
    face_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            face_emotions.append(analysis[0]['dominant_emotion'])
        except:
            pass

    cap.release()
    face_emotion = max(set(face_emotions), key=face_emotions.count) if face_emotions else "neutral"

    # ================== NEGATION + KEYWORD LOGIC ==================
    negation_words = ["not", "no", "never", "dont", "don't", "isn't", "wasn't", "aren't", "weren't"]

    positive_keywords = ["good", "great", "awesome", "amazing", "love", "excellent", "nice"]
    negative_keywords = ["bad", "worst", "boring", "hate", "poor", "terrible", "problem", "flaw"]

    detected_positive = []
    detected_negative = []

    words = text_lower.split()

    for i, word in enumerate(words):
        if word in positive_keywords:
            if i > 0 and words[i-1] in negation_words:
                detected_negative.append("not " + word)
            else:
                detected_positive.append(word)

        if word in negative_keywords:
            detected_negative.append(word)

    # ================== EXPLICIT NEGATIVE PHRASES ==================
    explicit_negative_phrases = [
        "don't like", "do not like", "not good", "not nice", "not worth",
        "not useful", "not great", "not satisfied", "don't want", "do not want"
    ]

    explicit_negative_detected = any(phrase in text_lower for phrase in explicit_negative_phrases)

    # ================== FINAL FUSION LOGIC ==================
    final = "Neutral"
    icon = "🟡"
    reason = "Mixed or neutral emotional signals detected"

    # 0️⃣ Explicit negative override
    if explicit_negative_detected:
        final = "Negative"
        icon = "🔴"
        reason = "Explicit negative phrase detected in speech (e.g., 'don't like', 'not good')"

    # 1️⃣ Text sentiment
    elif text_label == "NEGATIVE":
        final = "Negative"
        icon = "🔴"
        reason = "Negative sentiment detected in speech content"

    elif text_label == "POSITIVE":
        final = "Positive"
        icon = "🟢"
        reason = "Positive sentiment detected in speech content"

    # 2️⃣ Negation-aware keywords
    if detected_negative:
        final = "Negative"
        icon = "🔴"
        reason = f"Negative phrases detected: {detected_negative}"

    elif detected_positive:
        final = "Positive"
        icon = "🟢"
        reason = f"Positive words detected: {detected_positive}"

    # 3️⃣ Face emotion
    if face_emotion in ["angry", "sad", "disgust", "fear"]:
        final = "Negative"
        icon = "🔴"
        reason = f"Negative facial expression detected ({face_emotion})"

    elif face_emotion in ["happy", "surprise"]:
        final = "Positive"
        icon = "🟢"
        reason = f"Positive facial expression detected ({face_emotion})"

    # 4️⃣ Voice emotion (only if confident)
    if voice_conf > 0.4:
        if voice_emotion in ["angry", "sad", "fear", "disgust"]:
            final = "Negative"
            icon = "🔴"
            reason = f"Negative voice emotion detected ({voice_emotion})"
        elif voice_emotion == "happy":
            final = "Positive"
            icon = "🟢"
            reason = f"Positive voice emotion detected ({voice_emotion})"

    # 5️⃣ Energy fallback
    if energy < 0.02 and final == "Neutral":
        reason = "Low voice energy indicates hesitation or low engagement"

    # ================== FINAL OUTPUT ==================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## 🎯 Final Audience Response")
    st.markdown(f"### {icon} **{final.upper()}**")
    st.caption(reason)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================== ANALYTICS ==================
    with st.expander("📊 Emotional Breakdown & Reasoning"):
        st.markdown("### 🎤 Voice Analysis")
        st.write(f"• Emotion: {voice_emotion} ({voice_conf:.2f})")
        st.write(f"• Energy: {energy:.4f}")

        st.markdown("### 🧍 Face Analysis")
        st.write(f"• Dominant Emotion: {face_emotion}")

        st.markdown("### 🧾 Speech Analysis")
        st.write(f"• Transcribed Text: {text_output}")
        st.write(f"• Positive Keywords: {detected_positive}")
        st.write(f"• Negative Keywords: {detected_negative}")
        st.write(f"• Text Sentiment: {text_label} ({text_conf:.2f})")

        st.markdown("### 🧠 Fusion Reasoning")
        st.write(reason)

    # ================== AUDIENCE SUMMARY ==================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## 📝 Audience Feedback Summary")

    if final == "Positive":
        st.success("Audience shows strong interest and positive engagement. High likelihood of market acceptance.")
    elif final == "Negative":
        st.error("Audience shows dissatisfaction or concern. Product may face resistance in the market.")
    else:
        st.warning("Audience response is mixed or hesitant. Further refinement or clarification recommended.")

    st.markdown("</div>", unsafe_allow_html=True)
