import librosa
import numpy as np
import re

def extract_voice_features(audio_path, text):
    y, sr = librosa.load(audio_path, sr=None)

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))

    pitch = librosa.yin(y, fmin=60, fmax=300)
    pitch = pitch[pitch > 0]
    pitch_std = float(np.std(pitch)) if len(pitch) else 0.0

    duration = librosa.get_duration(y=y, sr=sr)
    speaking_rate = len(text.split()) / duration if duration > 0 else 0.0

    return {
        "energy_mean": energy_mean,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate
    }

def detect_hesitation(text):
    fillers = ["hmm", "uh", "um", "umm", "yeah"]
    t = text.lower()

    filler_count = sum(t.count(f) for f in fillers)
    pauses = len(re.findall(r"\.\.\.|,\s", text))

    total = filler_count + pauses

    if total >= 2:
        return "Medium"
    elif total == 1:
        return "Low"
    else:
        return "None"

def analyze_text_certainty(text):
    t = text.lower()

    certainty_words = [
        "definitely", "sure", "clearly", "absolutely",
        "very", "extremely", "love", "excellent"
    ]

    uncertainty_words = [
        "maybe", "i think", "not sure", "probably",
        "might", "could", "okay", "fine"
    ]

    return {
        "certainty": sum(t.count(w) for w in certainty_words),
        "uncertainty": sum(t.count(w) for w in uncertainty_words)
    }

def estimate_confidence(features, hesitation_level, text_certainty):
    score = 0

    if features["energy_mean"] > 0.06:
        score += 2
    elif features["energy_mean"] < 0.03:
        score -= 1

    if features["pitch_std"] < 45:
        score += 2
    else:
        score -= 1

    if 1.2 <= features["speaking_rate"] <= 3.0:
        score += 1

    if hesitation_level == "Medium":
        score -= 1

    score += text_certainty["certainty"] * 2
    score -= text_certainty["uncertainty"] * 2

    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"
