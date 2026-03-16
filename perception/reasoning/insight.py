import json
import nltk
from typing import List, Dict, Any, Optional

class TherapeuticInsight:  # Renamed to match main_live.py
    """Provides therapeutic insight analysis for user conversations."""
    
    TRIGGER_KEYWORDS = ['because', 'makes me', 'always', 'whenever', 'every time']
    NEGATIVE_EMOTIONS = ['sad', 'angry', 'frustrated', 'anxious', 'depressed', 'hopeless']
    
    def __init__(self):
        # Ensure NLTK resources are available for tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def identify_triggers(self, text: str) -> List[Dict[str, str]]:
        """Looks for keywords indicating cause-and-effect in user's speech."""
        text_lower = str(text or "").lower()
        triggers = []
        
        for keyword in self.TRIGGER_KEYWORDS:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                start = max(0, idx - 20)
                end = min(len(text_lower), idx + len(keyword) + 50)
                context = text_lower[start:end].strip()
                triggers.append({
                    'keyword': keyword,
                    'context': context
                })
        
        return triggers
    
    def detect_emotional_cycle(self, history: List[Any], current_emotion: str) -> Dict[str, Any]:
        """Checks if user is stuck in a negative emotion cycle."""
        recent_history = history[-5:] if len(history) >= 5 else history
        
        negative_count = 0
        emotions_found = []
        
        for entry in recent_history:
            # Handle both dict entries and raw strings
            emotion = ""
            if isinstance(entry, dict):
                emotion = str(entry.get('emotion', '')).lower()
            else:
                emotion = str(entry).lower()
            
            emotions_found.append(emotion)
            if emotion in self.NEGATIVE_EMOTIONS:
                negative_count += 1
        
        safe_current = str(current_emotion or "neutral").lower()
        if safe_current in self.NEGATIVE_EMOTIONS:
            negative_count += 1
            emotions_found.append(safe_current)
        
        is_stuck = negative_count > 3
        dominant = max(set(emotions_found), key=emotions_found.count) if emotions_found else "neutral"
        
        return {
            'is_stuck': is_stuck,
            'negative_count': negative_count,
            'recent_emotions': emotions_found,
            'dominant_emotion': dominant
        }
    
    def generate_strategy(self, sentiment_score: float) -> str:
        """Returns a therapeutic strategy based on sentiment polarity."""
        score = float(sentiment_score or 0.0)
        
        if score < -0.5:
            return 'Deep empathy & Crisis Validation'
        elif -0.5 <= score <= 0.2:
            return 'Cognitive Reflection & Insight'
        elif 0.2 < score <= 0.5:
            return 'Supportive encouragement'
        else:
            return 'Positive reinforcement'
    
    def analyze_situation(self, text: str, history: List[Any], current_emotion: str, sentiment_score: float) -> Dict[str, Any]:
        """Main entry point for the Reasoning layer."""
        triggers = self.identify_triggers(text)
        emotional_cycle = self.detect_emotional_cycle(history, current_emotion)
        strategy = self.generate_strategy(sentiment_score)
        
        # Determine if intervention is needed
        needs_intervention = bool(emotional_cycle['is_stuck'] or sentiment_score < -0.5)
        
        return {
            'triggers': triggers,
            'emotional_cycle': emotional_cycle,
            'recommended_strategy': strategy,
            'needs_intervention': needs_intervention,
            'summary': self._generate_summary(triggers, emotional_cycle, strategy)
        }
    
    def _generate_summary(self, triggers: List[Dict], emotional_cycle: Dict, strategy: str) -> str:
        """Generates a human-readable summary of the AGI's reasoning."""
        parts = []
        
        if triggers:
            parts.append(f"Identified {len(triggers)} trigger(s) in speech.")
        
        if emotional_cycle.get('is_stuck'):
            parts.append(f"User appears stuck in a negative cycle ({emotional_cycle['dominant_emotion']}).")
        
        parts.append(f"Recommended approach: {strategy}.")
        
        return ' '.join(parts) if parts else 'No significant patterns detected.'