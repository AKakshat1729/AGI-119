# core/ethics_personalization.py
# (Safe Version - Database Removed)

class EthicalAwarenessEngine:
    def detect_high_risk(self, text):
        keywords = ["suicide", "kill myself", "self harm", "die", "hurt myself", "end it all"]
        return any(k in text.lower() for k in keywords)

    def ethical_response(self):
        return (
            "I'm really sorry you're feeling this way, but I'm an AI and I can't provide the help you need right now. "
            "Please consider reaching out to a trusted person or a mental health professional immediately."
        )

class PersonalizationEngine:
    """
    Adjusts the tone style based on simple emotion labels.
    """
    def adapt_tone(self, emotion):
        tones = {
            "sad": "gentle and supportive",
            "angry": "calm and de-escalating",
            "happy": "positive and reinforcing",
            "confused": "clear and structured"
        }
        return tones.get(emotion, "neutral")

# REMOVED: RelationshipManager (It conflicted with your main LongTermMemory)
# REMOVED: EthicalPersonalizedAgent (We will build a better agent in app.py)