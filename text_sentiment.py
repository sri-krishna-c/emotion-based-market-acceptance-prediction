def classify_transcript_sentiment(text):
    t = text.lower()

    positive_words = ["like", "love", "good", "great", "excellent", "amazing"]
    negative_words = ["bad", "broken", "boring", "hate", "problem", "issue"]
    downgrade_words = ["okay", "fine", "alright", "not bad"]

    pos = sum(t.count(w) for w in positive_words)
    neg = sum(t.count(w) for w in negative_words)
    down = sum(t.count(w) for w in downgrade_words)

    has_but = any(w in t for w in [" but ", " however ", " though "])

    if pos > 0 and neg > 0:
        sentiment = "Neutral"
    elif pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    if sentiment == "Positive" and (down > 0 or has_but):
        sentiment = "Neutral"

    return sentiment, pos, neg, down, has_but
