import re

POSITIVE = ["good", "great", "excellent", "amazing", "fascinating", "love"]
NEGATIVE = ["bad", "worst", "hate", "boring", "not good", "don't like"]

UNCERTAIN = ["maybe", "i think", "i guess", "not sure", "probably"]
CERTAIN = ["definitely", "clearly", "absolutely", "very much"]

def analyze_text(text):
    t = text.lower()

    negation = bool(re.search(r"\b(not|don't|never|no)\b", t))

    pos_hits = sum(w in t for w in POSITIVE)
    neg_hits = sum(w in t for w in NEGATIVE)
    uncertain_hits = sum(w in t for w in UNCERTAIN)
    certain_hits = sum(w in t for w in CERTAIN)

    sentiment = "Neutral"
    if neg_hits > pos_hits:
        sentiment = "Negative"
    elif pos_hits > neg_hits:
        sentiment = "Positive"

    if negation and pos_hits > 0:
        sentiment = "Negative"

    return {
        "sentiment": sentiment,
        "certainty_score": certain_hits,
        "uncertainty_score": uncertain_hits,
        "negation": negation
    }
