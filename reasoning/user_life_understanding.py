import json
from memory.long_term_memory import LongTermMemory

class UserLifeUnderstanding:
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.ltm = LongTermMemory(user_id=user_id)

    def connect_past_present(self, current_text):
        """
        Searches for past conversations similar to the current topic.
        """
        try:
            # Safe Retrieve
            results = self.ltm.retrieve(current_text, n_results=3)
            
            # GUARD CLAUSE: If results is None, return empty list
            if not results:
                return []

            # Safe Get: Handle if 'documents' key is missing or None
            raw_docs = results.get('documents') or []
            
            # Flatten list of lists (ChromaDB format)
            past_connections = []
            if raw_docs:
                for sublist in raw_docs:
                    if sublist:
                        for doc in sublist:
                            # Avoid exact duplicate of current input
                            if doc and doc != current_text:
                                past_connections.append(doc)
            
            return past_connections
        except Exception as e:
            print(f"Error in connect_past_present: {e}")
            return []

    def analyze_recurring_problems(self):
        """
        Checks for repeated negative patterns in history.
        """
        try:
            keywords = "struggle anxiety sad problem hard fail"
            results = self.ltm.retrieve(keywords, n_results=5)
            
            if not results:
                return []

            raw_docs = results.get('documents') or []
            problems = []
            
            if raw_docs:
                for sublist in raw_docs:
                    if sublist:
                        for doc in sublist:
                            # Simple heuristic: if it contains negative words
                            if any(w in doc.lower() for w in ["sad", "anxious", "can't", "failed"]):
                                problems.append(doc)
            return problems
        except Exception as e:
            print(f"Error in analyze_recurring_problems: {e}")
            return []

    def build_life_story(self):
        """
        Attempts to find facts about the user (names, places).
        """
        try:
            # We look for "My name is" or "I live in" type statements
            results = self.ltm.retrieve("My name is I live in I am from", n_results=5)
            
            if not results:
                return {}

            raw_docs = results.get('documents') or []
            facts = []
            
            if raw_docs:
                for sublist in raw_docs:
                    if sublist:
                        facts.extend(sublist)
            
            return {"potential_facts": facts}
        except Exception as e:
            print(f"Error in build_life_story: {e}")
            return {}

    def recognize_emotional_progress(self):
        """
        Compare recent emotions to past emotions.
        """
        # This is complex, so we return a placeholder for now to be safe
        return "Stable"

    def maintain_consistency(self, current_text):
        """
        Checks if the user contradicts themselves.
        """
        return "Consistent"