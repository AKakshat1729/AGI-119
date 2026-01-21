def detect_emotion(text):
    # Simple emotion detection based on keywords
    text_lower = text.lower()
    if any(word in text_lower for word in ['happy', 'joy', 'great', 'awesome']):
        return 'happy'
    elif any(word in text_lower for word in ['sad', 'depressed', 'unhappy', 'bad']):
        return 'sad'
    elif any(word in text_lower for word in ['angry', 'mad', 'furious']):
        return 'angry'
    elif any(word in text_lower for word in ['scared', 'afraid', 'worried']):
        return 'fear'
    else:
        return 'neutral'