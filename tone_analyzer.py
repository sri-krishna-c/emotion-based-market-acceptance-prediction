def analyze_tone(text, voice):
    t = text.lower()
    fillers = ["hmm", "yeah", "uh", "um"]

    detected = [f for f in fillers if f in t]

    if voice["energy_mean"] > 0.08 and voice["pitch_std"] > 55:
        tone = "Excited"
    elif voice["energy_mean"] < 0.04 and voice["pitch_std"] < 40:
        tone = "Flat"
    elif detected:
        tone = "Hesitant"
    else:
        tone = "Calm"

    return {
        "tone": tone,
        "fillers_detected": detected
    }
