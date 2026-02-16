def estimate_confidence(voice, text, sarcasm):
    score = 0

    # Energy (NOT loudness)
    if voice["energy"] > 0.04:
        score += 2
    elif voice["energy"] < 0.02:
        score -= 2

    # Pitch stability
    if voice["pitch_std"] < 40:
        score += 2
    else:
        score -= 1

    # Text certainty
    score += text["certainty_score"] * 2
    score -= text["uncertainty_score"] * 2

    # Negation penalty
    if text["negation"]:
        score -= 2

    # Sarcasm destroys confidence
    if sarcasm["sarcasm"]:
        score -= 3

    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"
