import librosa
import numpy as np

def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = pitch[pitch > 0]

    pitch_mean = float(np.mean(pitch)) if len(pitch) else 0
    pitch_std = float(np.std(pitch)) if len(pitch) else 0

    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "energy": energy_mean,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "duration": duration
    }
