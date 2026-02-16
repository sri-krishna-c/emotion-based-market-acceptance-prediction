def analyze_nuance(text, voice, text_info):
    reasons = []

    hesitation = text.count("uh") + text.count("um") + text.count("hmm")

    authenticity = "High"
    if voice["energy_mean"] < 0.03 or text_info["uncertainty"] > 0:
        authenticity = "Low"
        reasons.append("Low energy or uncertain language")

    cognitive_load = "Low"
    if voice["speaking_rate"] < 1.2:
        cognitive_load = "High"
        reasons.append("Slow speaking rate")

    politeness = False
    if text_info["sentiment"] == "Positive" and authenticity == "Low":
        politeness = True
        reasons.append("Likely polite approval")

    return {
        "hesitation_count": hesitation,
        "authenticity": authenticity,
        "cognitive_load": cognitive_load,
        "politeness": politeness,
        "reasons": reasons
    }
