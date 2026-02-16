def detect_sarcasm(text, energy_mean, pitch_std, fillers_detected):
    t = text.lower()

    positive_words = ["good", "great", "amazing", "nice", "excellent"]

    positive_lexical = any(w in t for w in positive_words)
    flat_voice = energy_mean < 0.035
    compressed_pitch = pitch_std < 35
    filler_based = "yeah" in fillers_detected or "hmm" in fillers_detected

    sarcasm = positive_lexical and flat_voice and compressed_pitch and filler_based

    if sarcasm:
        return {
            "sarcasm": True,
            "reason": "Positive words spoken with flat tone and hesitation"
        }
    else:
        return {
            "sarcasm": False,
            "reason": "No sarcasm detected"
        }
