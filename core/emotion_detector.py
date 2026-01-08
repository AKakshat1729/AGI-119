def detect_emotion(text):
    text = text.lower()
    if "sad" in text:
        return "sad"
    if "angry" in text:
        return "angry"
    if "happy" in text:
        return "happy"
    return "neutral"
