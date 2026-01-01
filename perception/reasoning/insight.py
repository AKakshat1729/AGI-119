import json
from textblob import TextBlob
from memory.long_term_memory import LongTermMemory

class InsightGenerator:
    """
    Provides therapeutic insight analysis for user conversations.
    Wraps 'TherapeuticInsight' logic to work with the main app.
    """
    
    TRIGGER_KEYWORDS = ['because', 'makes me', 'always', 'whenever', 'every time']
    NEGATIVE_EMOTIONS = ['sad', 'angry', 'frustrated', 'anxious', 'depressed', 'hopeless']
    
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.ltm = LongTermMemory(user_id=user_id)

    def generate(self, transcript):
        """
        Main entry point called by app.py.
        Automatically fetches history and sentiment to perform analysis.
        """
        # 1. Calculate Sentiment internally
        blob = TextBlob(transcript)
        
        # ✅ THE FIX: Safely get polarity using getattr
        # This tells Python: "Get 'polarity', but if you can't find it, assume 0.0"
        sentiment_score = getattr(blob.sentiment, 'polarity', 0.0)
        
        # 2. Determine basic emotion (simplified for this module)
        current_emotion = "neutral"
        if sentiment_score < -0.1: current_emotion = "sad"
        elif sentiment_score > 0.1: current_emotion = "happy"

        # 3. Fetch History safely
        history_docs = self._get_safe_history(transcript)

        # 4. Run the core analysis
        return self.analyze_situation(transcript, history_docs, current_emotion, sentiment_score)
    def _get_safe_history(self, text):
        """Helper to get history without crashing on lists."""
        try:
            results = self.ltm.retrieve(text, n_results=5)
            docs = results.get('documents') or []
            # Flatten if nested list
            if docs and isinstance(docs[0], list):
                return [str(item) for sublist in docs for item in sublist]
            return [str(d) for d in docs]
        except:
            return []

    # --- Your Original Logic (Renamed/Adapted) ---

    def identify_triggers(self, text: str) -> list:
        text_lower = text.lower()
        triggers = []
        for keyword in self.TRIGGER_KEYWORDS:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(keyword) + 50)
                context = text[start:end].strip()
                triggers.append({'keyword': keyword, 'context': context})
        return triggers
    
    def detect_emotional_cycle(self, history: list, current_emotion: str) -> dict:
        recent_history = history[-5:] if len(history) >= 5 else history
        negative_count = 0
        emotions_found = []
        
        # Simple emotion extraction from string history
        for entry in recent_history:
            # Assume entry is a string; check for negative keywords
            entry_str = str(entry).lower()
            found_neg = False
            for neg in self.NEGATIVE_EMOTIONS:
                if neg in entry_str:
                    negative_count += 1
                    emotions_found.append(neg)
                    found_neg = True
                    break
            if not found_neg:
                emotions_found.append("neutral")
        
        if current_emotion in self.NEGATIVE_EMOTIONS:
            negative_count += 1
        
        return {
            'is_stuck': negative_count >= 3,
            'negative_count': negative_count,
            'dominant_emotion': max(set(emotions_found), key=emotions_found.count) if emotions_found else "neutral"
        }
    
    def generate_strategy(self, sentiment_score: float) -> str:
        if sentiment_score < -0.5: return 'Deep empathy'
        elif -0.5 <= sentiment_score <= 0.2: return 'Insight/Reflection'
        elif 0.2 < sentiment_score <= 0.5: return 'Supportive encouragement'
        else: return 'Positive reinforcement'
    
    def analyze_situation(self, text: str, history: list, current_emotion: str, sentiment_score: float) -> dict:
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
        parts = []
        if triggers:
            parts.append(f"Identified {len(triggers)} trigger(s).")
        if emotional_cycle['is_stuck']:
            parts.append(f"User stuck in {emotional_cycle['dominant_emotion']} cycle.")
        parts.append(f"Strategy: {strategy}.")
        return ' '.join(parts) if parts else 'No significant patterns.'