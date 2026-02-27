import json
from textblob import TextBlob
from memory.long_term_memory import LongTermMemory
import nltk
from collections import Counter
from datetime import datetime
# Ensure NLTK data is downloaded (run once if needed)
# nltk.download('punkt')
# nltk.download('stopwords')

class UserLifeUnderstanding:
    def __init__(self, user_id="default", memory_store=None):
        """
        Initializes the UserLifeUnderstanding with a memory store.
        """
        self.user_id = user_id
        if memory_store:
            self.memory_store = memory_store
        else:
            from api.memory_store import ServerMemoryStore
            self.memory_store = ServerMemoryStore()

    def connect_past_present(self, current_input, n_results=5):
        """
        Connects past conversations with the present by retrieving relevant past entries.
        Args:
            current_input (str): The current user input (transcript).
            n_results (int): Number of past results to retrieve.
        Returns:
            list: Relevant past conversations.
        """
        results = self.memory_store.retrieve_memories(user_id=self.user_id, query=current_input, top_k=n_results)
        return [r['text'] for r in results]

    def analyze_recurring_problems(self, n_recent=50):
        """
        Analyzes recurring problems and emotional patterns by examining stored memories.
        Args:
            n_recent (int): Number of recent entries to analyze.
        Returns:
            dict: Analysis of recurring problems and patterns.
        """
        # Retrieve recent memories from logs (conversations) or episodic
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_recent)
        if not mems:
             mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="episodic", top_k=n_recent)
        
        problems = []
        emotions = []
        for mem in mems:
            doc = mem['text']
            try:
                # Try JSON first
                data = json.loads(doc)
                transcript = data.get('transcript', doc)
            except:
                transcript = doc
            
            sentiment = TextBlob(transcript).sentiment.polarity
            emotions.append(sentiment)
            # Simple keyword extraction for problems (negative words)
            if sentiment < 0:
                words = nltk.word_tokenize(transcript.lower())
                problems.extend([w for w in words if w not in nltk.corpus.stopwords.words('english') and len(w) > 3])
        
        recurring_problems = Counter(problems).most_common(10)
        emotional_patterns = {
            'average_sentiment': sum(emotions) / len(emotions) if emotions else 0,
            'sentiment_variance': sum((x - (sum(emotions)/len(emotions)))**2 for x in emotions) / len(emotions) if emotions else 0
        }
        return {
            'recurring_problems': recurring_problems,
            'emotional_patterns': emotional_patterns
        }

    def build_life_story(self, n_entries=100):
        """
        Builds a long-term understanding of the user's life story by summarizing stored memories.
        Args:
            n_entries (int): Number of entries to summarize.
        Returns:
            str: Summary of the user's life story.
        """
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="episodic", top_k=n_entries)
        if not mems:
            mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_entries)
            
        summaries = []
        for mem in mems:
            doc = mem['text']
            try:
                data = json.loads(doc)
                transcript = data.get('transcript', doc)
                summaries.append(transcript)
            except:
                summaries.append(doc)
        
        if not summaries:
            return "No life history recorded yet."

        # Simple concatenation and summarization
        full_text = ' '.join(summaries)
        blob = TextBlob(full_text)
        sentences = blob.sentences
        # Take first and last few sentences as summary
        summary = ' '.join([str(s) for s in sentences[:3] + sentences[-3:]])
        return summary

    def recognize_emotional_progress(self, n_entries=50):
        """
        Recognizes emotional progress or setbacks by tracking sentiment over time.
        Args:
            n_entries (int): Number of entries to analyze.
        Returns:
            dict: Emotional progress analysis.
        """
        # Retrieve from conversation logs
        mems = self.memory_store.retrieve_memories(user_id=self.user_id, query="", memory_type="conversation", top_k=n_entries)
        if not mems:
            return {'progress': 'No data'}
            
        sentiments = []
        for mem in mems:
            doc = mem['text']
            # Simple sentiment analysis on the raw text
            sentiment = TextBlob(doc).sentiment.polarity
            sentiments.append(sentiment)
            
        if not sentiments:
            return {'progress': 'No data'}
            
        # Compare first half with second half (reverse chronological)
        # mems are usually reverse chronological
        recent_half = sentiments[:len(sentiments)//2]
        older_half = sentiments[len(sentiments)//2:]
        
        initial_avg = sum(older_half) / len(older_half) if older_half else 0
        recent_avg = sum(recent_half) / len(recent_half) if recent_half else 0
        
        progress = 'improving' if recent_avg > initial_avg + 0.05 else 'declining' if recent_avg < initial_avg - 0.05 else 'stable'
        
        return {
            'initial_sentiment': initial_avg,
            'recent_sentiment': recent_avg,
            'progress': progress
        }

    def maintain_consistency(self, current_input):
        """
        Maintains consistency across sessions by retrieving relevant past context.
        Args:
            current_input (str): Current input.
        Returns:
            str: Consistent context summary.
        """
        past = self.connect_past_present(current_input, n_results=3)
        context = ' '.join(past)
        return context