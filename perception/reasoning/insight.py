class TheraputicInsight:
    """Provides therapeutic insight analysis for user conversations."""
    
    TRIGGER_KEYWORDS = ['because', 'makes me', 'always', 'whenever', 'every time']
    NEGATIVE_EMOTIONS = ['sad', 'angry', 'frustrated', 'anxious', 'depressed', 'hopeless']
    
    def identify_triggers(self, text: str) -> list:
        """
        Looks for keywords indicating cause-and-effect in user's speech.
        Returns list of identified trigger phrases.
        """
        text_lower = text.lower()
        triggers = []
        
        for keyword in self.TRIGGER_KEYWORDS:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(keyword) + 50)
                context = text[start:end].strip()
                triggers.append({
                    'keyword': keyword,
                    'context': context
                })
        
        return triggers
    
    def detect_emotional_cycle(self, history: list, current_emotion: str) -> dict:
        """
        Checks the last 5 turns in history to detect if user is stuck
        in a negative emotion for more than 3 turns.
        """
        recent_history = history[-5:] if len(history) >= 5 else history
        
        negative_count = 0
        emotions_found = []
        
        for entry in recent_history:
            emotion = entry.get('emotion', '').lower() if isinstance(entry, dict) else str(entry).lower()
            emotions_found.append(emotion)
            if emotion in self.NEGATIVE_EMOTIONS:
                negative_count += 1
        
        if current_emotion.lower() in self.NEGATIVE_EMOTIONS:
            negative_count += 1
            emotions_found.append(current_emotion.lower())
        
        is_stuck = negative_count > 3
        
        return {
            'is_stuck': is_stuck,
            'negative_count': negative_count,
            'recent_emotions': emotions_found,
            'dominant_emotion': max(set(emotions_found), key=emotions_found.count) if emotions_found else None
        }
    
    def generate_strategy(self, sentiment_score: float) -> str:
        """
        Returns a strategy string based on sentiment score.
        Score typically ranges from -1 (very negative) to 1 (very positive).
        """
        if sentiment_score < -0.5:
            return 'Deep empathy'
        elif -0.5 <= sentiment_score <= 0.2:
            return 'Insight/Reflection'
        elif 0.2 < sentiment_score <= 0.5:
            return 'Supportive encouragement'
        else:
            return 'Positive reinforcement'
    
    def analyze_situation(self, text: str, history: list, current_emotion: str, sentiment_score: float) -> dict:
        """
        Main method that calls all three analysis methods and returns a summary dictionary.
        """
        triggers = self.identify_triggers(text)
        emotional_cycle = self.detect_emotional_cycle(history, current_emotion)
        strategy = self.generate_strategy(sentiment_score)
        
        return {
            'triggers': triggers,
            'emotional_cycle': emotional_cycle,
            'recommended_strategy': strategy,
            'needs_intervention': emotional_cycle['is_stuck'] or sentiment_score < -0.5,
            'summary': self._generate_summary(triggers, emotional_cycle, strategy)
        }
    
    def _generate_summary(self, triggers: list, emotional_cycle: dict, strategy: str) -> str:
        """Generates a human-readable summary of the analysis."""
        parts = []
        
        if triggers:
            parts.append(f"Identified {len(triggers)} trigger(s) in speech.")
        
        if emotional_cycle['is_stuck']:
            parts.append(f"User appears stuck in negative emotional cycle ({emotional_cycle['dominant_emotion']}).")
        
        parts.append(f"Recommended approach: {strategy}.")
        
        return ' '.join(parts) if parts else 'No significant patterns detected.'
