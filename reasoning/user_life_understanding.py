import json
from textblob import TextBlob
from memory.long_term_memory import LongTermMemory
from typing import Any

class UserLifeUnderstanding:
    def __init__(self, user_id="default"):
        self.ltm = LongTermMemory(user_id=user_id)

    def build_life_story(self):
        """Builds a summary of the user's life based on all LTM entries."""
        data = self.ltm.get_all()
        # FIX: Safely get documents and handle None with 'or []'
        documents = data.get('documents') or []
        
        # Flatten documents if they are nested (List of Lists)
        if documents and isinstance(documents[0], list):
            flat_docs = [str(item) for sublist in documents for item in sublist]
        else:
            flat_docs = [str(doc) for doc in documents]

        # FIX: Access TextBlob sentiment safely to avoid 'cached_property' error
        positive_points = []
        for doc in flat_docs:
            blob = TextBlob(doc)
            # We access .sentiment once and store it. 
            # This helps the type checker understand what 'sentiment' is.
            sentiment = blob.sentiment 
            if getattr(sentiment, 'polarity', 0) > 0.1:
                positive_points.append(doc)
        
        return {"story_segments": len(flat_docs), "positivity_count": len(positive_points)}

    def analyze_recurring_problems(self):
        """Identifies patterns or recurring issues mentioned by the user."""
        data = self.ltm.get_all()
        # FIX: Object of type "None" cannot be used as iterable
        documents = data.get('documents') or []
        
        if not documents:
            return []

        # Flatten logic
        if isinstance(documents[0], list):
            flat_docs = [str(item) for sublist in documents for item in sublist]
        else:
            flat_docs = [str(doc) for doc in documents]

        problems = []
        keywords = ["problem", "issue", "struggle", "difficult", "hard"]
        
        for doc in flat_docs:
            if any(word in doc.lower() for word in keywords):
                problems.append(doc)
        
        return problems

    def recognize_emotional_progress(self):
        """Tracks if sentiment is improving over time."""
        data = self.ltm.get_all()
        documents = data.get('documents') or []
        
        if len(documents) < 2:
            return "Not enough data to track progress."

        # Flatten logic to ensure we have a list of strings
        if isinstance(documents[0], list):
            flat_docs = [str(item) for sublist in documents for item in sublist]
        else:
            flat_docs = [str(doc) for doc in documents]

        # --- THE FIX FOR LINE 75 ---
        # We cast the sentiment to 'Any' to stop Pylance from complaining 
        # about the 'cached_property' internal attribute.
        first_sentiment: Any = TextBlob(flat_docs[0]).sentiment
        last_sentiment: Any = TextBlob(flat_docs[-1]).sentiment
        
        # Now Pylance will allow access to .polarity without errors
        if last_sentiment.polarity > first_sentiment.polarity:
            return "Showing emotional improvement."
            
        return "Stable emotional state."
    def connect_past_present(self, current_input, n_results=5):
        """Retrieves flat list of strings from LTM."""
        results = self.ltm.retrieve(current_input, n_results=n_results)
        documents = results.get('documents') or []
        
        if documents and isinstance(documents[0], list):
            return [str(item) for sublist in documents for item in sublist]
        return [str(doc) for doc in documents]

    def maintain_consistency(self, current_input):
        """Maintains session consistency."""
        past = self.connect_past_present(current_input, n_results=3)
        # Fix: Ensure past is treated as a list even if connect_past_present fails
        context = ' '.join(past or [])
        return context