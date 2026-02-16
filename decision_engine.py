def final_decision(text_info, nuance, sarcasm):
    if sarcasm["sarcasm"]:
        return "Negative", "Sarcasm detected"

    if text_info["sentiment"] == "Negative":
        return "Negative", "Negative wording"

    if text_info["sentiment"] == "Positive":
        if nuance["authenticity"] == "High":
            return "Positive", "Genuine positive response"
        else:
            return "Neutral", "Polite or low-conviction approval"

    return "Neutral", "Uncertain or mixed response"
