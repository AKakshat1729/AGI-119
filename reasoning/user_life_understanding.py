import json
from typing import Any, List, Dict, Optional
from collections import Counter
from datetime import datetime
import nltk
from textblob import TextBlob

# --- Fallback Integration ---
try:
    from memory.long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

class UserLifeUnderstanding:
    def __init__(self, user_id: str = "default", memory_store: Any = None):
        """
        Initializes the UserLifeUnderstanding with a memory store.
        """
        self.user_id = user_id
        if memory_store:
            self.memory_store = memory_store
        else:
            # Lazy import to avoid circular dependencies
            from api.memory_store import ServerMemoryStore
            self.memory_store = ServerMemoryStore()

    def connect_past_present(self, current_input: str, n_results: int = 5) -> List[str]:
        """
        Connects past conversations with the present by retrieving relevant past entries.
        """
        results = self.memory_store.retrieve_memories(user_id=self.user_id, query=current_input, top_k=n_results)
        return [str(r.get('text', '')) for r in results]

    def analyze_recurring_problems(self, n_recent: int = 50) -> Dict[str, Any]:
        """
        Analyzes recurring problems and emotional patterns by examining stored memories.
        """
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_recent)
        if not mems:
             mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="episodic", top_k=n_recent)
        
        problems: List[str] = []
        emotions: List[float] = []
        
        # Stopwords preparation
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            stop_words = set()

        for mem in mems:
            doc = str(mem.get('text', ''))
            try:
                data = json.loads(doc)
                transcript = str(data.get('transcript', doc))
            except:
                transcript = doc
            
            # PYLANCE FIX: Cast sentiment to Any to access .polarity
            blob = TextBlob(transcript)
            sentiment_obj: Any = blob.sentiment
            sentiment = float(sentiment_obj.polarity)
            
            emotions.append(sentiment)
            
            if sentiment < 0:
                words = nltk.word_tokenize(transcript.lower())
                problems.extend([w for w in words if w not in stop_words and len(w) > 3])
        
        avg_sentiment = sum(emotions) / len(emotions) if emotions else 0.0
        
        return {
            'recurring_problems': Counter(problems).most_common(10),
            'emotional_patterns': {
                'average_sentiment': round(avg_sentiment, 4),
                'sentiment_variance': sum((x - avg_sentiment)**2 for x in emotions) / len(emotions) if emotions else 0.0
            }
        }

    def build_life_story(self, n_entries: int = 100) -> str:
        """
        Builds a long-term understanding of the user's life story.
        """
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="episodic", top_k=n_entries)
        if not mems:
            mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_entries)
            
        summaries: List[str] = []
        for mem in mems:
            doc = str(mem.get('text', ''))
            try:
                data = json.loads(doc)
                summaries.append(str(data.get('transcript', doc)))
            except:
                summaries.append(doc)
        
        if not summaries:
            return "No life history recorded yet."

        full_text = ' '.join(summaries)
        blob = TextBlob(full_text)
        
        # PYLANCE FIX: Cast sentences to Any to allow slicing/indexing
        sentences_list: Any = blob.sentences
        sentences = list(sentences_list)
        
        if len(sentences) <= 6:
            return str(blob)
            
        summary = ' '.join([str(s) for s in sentences[:3] + sentences[-3:]])
        return summary

    def recognize_emotional_progress(self, n_entries: int = 50) -> Dict[str, Any]:
        """
        Recognizes emotional progress or setbacks by tracking sentiment over time.
        """
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_entries)
        if not mems:
            return {'progress': 'No data'}
            
        sentiments: List[float] = []
        for mem in mems:
            doc = str(mem.get('text', ''))
            # PYLANCE FIX: Cast sentiment to Any
            blob = TextBlob(doc)
            sentiment_obj: Any = blob.sentiment
            sentiments.append(float(sentiment_obj.polarity))
            
        if not sentiments:
            return {'progress': 'No data'}
            
        # Divide into recent and older halves
        mid = len(sentiments) // 2
        recent_half = sentiments[:mid]
        older_half = sentiments[mid:]
        
        initial_avg = sum(older_half) / len(older_half) if older_half else 0.0
        recent_avg = sum(recent_half) / len(recent_half) if recent_half else 0.0
        
        progress = 'stable'
        if recent_avg > initial_avg + 0.05:
            progress = 'improving'
        elif recent_avg < initial_avg - 0.05:
            progress = 'declining'
        
        return {
            'initial_sentiment': round(initial_avg, 4),
            'recent_sentiment': round(recent_avg, 4),
            'progress': progress
        }

    def maintain_consistency(self, current_input: str) -> str:
        """
        Maintains consistency across sessions by retrieving relevant past context.
        """
        past = self.connect_past_present(current_input, n_results=3)
        return ' '.join(past)