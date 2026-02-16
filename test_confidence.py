import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper

from voice_confidence import extract_voice_features, detect_hesitation, analyze_text_certainty
from sarcasm_detector import detect_sarcasm

# =========================
# STEP 1: RECORD AUDIO
# =========================

DURATION = 5
SR = 44100

print("\n🎙 Recording... Speak naturally")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()

audio = audio / (np.max(np.abs(audio)) + 1e-6)
sf.write("input.wav", audio, SR)
print("✅ Recording saved")

# =========================
# STEP 2: TRANSCRIBE
# =========================

model = whisper.load_model("small")
result = model.transcribe("input.wav", fp16=False)
text = result["text"].strip()

print("\n🧾 Transcribed Text:", text)

# =========================
# STEP 3: VOICE FEATURES
# =========================

features = extract_voice_features("input.wav", text)
certainty = analyze_text_certainty(text)

fillers = ["yeah", "hmm", "uh", "umm", "um"]
fillers_detected = [f for f in fillers if f in text.lower()]

sarcasm_info = detect_sarcasm(
    text,
    features["energy_mean"],
    features["pitch_std"],
    fillers_detected
)

# =========================
# FINAL OUTPUT
# =========================

print("\n================= ANALYSIS RESULT =================")
print("Transcript       :", text)
print("Energy Mean      :", round(features["energy_mean"], 4))
print("Pitch Std        :", round(features["pitch_std"], 2))
print("Speaking Rate    :", round(features["speaking_rate"], 2))
print("Fillers Detected :", fillers_detected)
print("Sarcasm Detected :", sarcasm_info["sarcasm"])
print("Sarcasm Reason   :", sarcasm_info["reason"])
print("==================================================")
