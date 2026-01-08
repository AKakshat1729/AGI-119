from datetime import datetime
from core.memory_store import save_memory

class EthicalAwarenessEngine:
    def detect_high_risk(self, text):
        keywords = ["suicide", "kill myself", "self harm", "die"]
        return any(k in text.lower() for k in keywords)

    def ethical_response(self):
        return (
            "I'm really sorry you're feeling this way. "
            "Please consider reaching out to a trusted person "
            "or a mental health professional."
        )

class PersonalizationEngine:
    def adapt_tone(self, emotion):
        tones = {
            "sad": "gentle",
            "angry": "calm",
            "happy": "positive",
            "confused": "clear"
        }
        return tones.get(emotion, "neutral")

class RelationshipManager:
    def log(self, text, emotion):
        save_memory(text, emotion)

class EthicalPersonalizedAgent:
    def __init__(self):
        self.ethics = EthicalAwarenessEngine()
        self.personalization = PersonalizationEngine()
        self.relationship = RelationshipManager()

    def generate_response(self, user_text, emotion):
        if self.ethics.detect_high_risk(user_text):
            return self.ethics.ethical_response()

        tone = self.personalization.adapt_tone(emotion)
        self.relationship.log(user_text, emotion)

        return f"I understand you. I'll respond in a {tone} way. Tell me more."
