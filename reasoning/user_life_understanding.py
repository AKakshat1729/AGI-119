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
    def __init__(self, user_id="default"):
        """
        Initializes the UserLifeUnderstanding with LongTermMemory for the user.
        """
        self.user_id = user_id
        self.ltm = LongTermMemory(user_id=user_id)

    def connect_past_present(self, current_input, n_results=5):
        """
        Connects past conversations with the present by retrieving relevant past entries.
        Args:
            current_input (str): The current user input (transcript).
            n_results (int): Number of past results to retrieve.
        Returns:
            list: Relevant past conversations.
        """
        results = self.ltm.retrieve(current_input, n_results=n_results)
        return results['documents'] if 'documents' in results else []

    def analyze_recurring_problems(self, n_recent=50):
        """
        Analyzes recurring problems and emotional patterns by examining stored memories.
        Args:
            n_recent (int): Number of recent entries to analyze.
        Returns:
            dict: Analysis of recurring problems and patterns.
        """
        # Retrieve recent memories
        all_docs = self.ltm.collection.get(limit=n_recent)['documents']
        problems = []
        emotions = []
        for doc in all_docs:
            try:
                data = json.loads(doc)
                # Assume data has 'transcript' and 'tone'
                transcript = data.get('transcript', '')
                tone = data.get('tone', {})
                sentiment = TextBlob(transcript).sentiment.polarity
                emotions.append(sentiment)
                # Simple keyword extraction for problems (negative words)
                if sentiment < 0:
                    words = nltk.word_tokenize(transcript.lower())
                    problems.extend([w for w in words if w not in nltk.corpus.stopwords.words('english')])
            except:
                continue
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
        all_docs = self.ltm.collection.get(limit=n_entries)['documents']
        summaries = []
        for doc in all_docs:
            try:
                data = json.loads(doc)
                transcript = data.get('transcript', '')
                summaries.append(transcript)
            except:
                summaries.append(doc)
        # Simple concatenation and summarization (in real, use better summarizer)
        full_text = ' '.join(summaries)
        # For now, return a basic summary
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
        all_docs = self.ltm.collection.get(limit=n_entries)['documents']
        sentiments = []
        for doc in all_docs:
            try:
                data = json.loads(doc)
                transcript = data.get('transcript', '')
                sentiment = TextBlob(transcript).sentiment.polarity
                sentiments.append(sentiment)
            except:
                continue
        if not sentiments:
            return {'progress': 'No data'}
        initial_avg = sum(sentiments[:len(sentiments)//2]) / (len(sentiments)//2) if sentiments else 0
        recent_avg = sum(sentiments[len(sentiments)//2:]) / (len(sentiments) - len(sentiments)//2) if sentiments else 0
        progress = 'improving' if recent_avg > initial_avg else 'declining' if recent_avg < initial_avg else 'stable'
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