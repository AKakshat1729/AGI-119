import json
from textblob import TextBlob
from memory.long_term_memory import LongTermMemory
from memory.working_memory import WorkingMemory
import nltk
from collections import Counter

class InternalCognition:
    def __init__(self, user_id="default"):
        """
        Initializes the Internal Cognition module for self-reflection.
        """
        self.user_id = user_id
        self.ltm = LongTermMemory(user_id=user_id)
        self.wm = WorkingMemory()

    def deep_semantic_understanding(self, transcript, nlu_result, tone):
        """
        Performs deep semantic understanding by combining transcript, NLU, and tone analysis.
        """
        # Extract key elements
        entities = nlu_result.get('entities', [])
        semantic_roles = nlu_result.get('semantic_roles', [])
        emotions = tone.get('emotions', [])
        sentiment = tone.get('sentiment', {})

        # Analyze context from long-term memory
        past_context = self.ltm.retrieve(transcript, n_results=3)
        past_docs = past_context.get('documents', []) if past_context else []

        # Identify patterns and deeper meanings
        deeper_meaning = self._analyze_deeper_meaning(transcript, entities, semantic_roles, emotions, past_docs)

        # Determine intent beyond surface level
        true_intent = self._infer_true_intent(transcript, nlu_result, tone, past_docs)

        return {
            "deeper_meaning": deeper_meaning,
            "true_intent": true_intent,
            "context_relevance": len(past_docs) > 0,
            "confidence": self._calculate_understanding_confidence(nlu_result, tone)
        }

    def _analyze_deeper_meaning(self, transcript, entities, semantic_roles, emotions, past_docs):
        """
        Analyzes deeper meaning from various components.
        """
        meanings = []

        # Check for emotional subtext
        if emotions:
            meanings.append(f"Emotional undertone: {', '.join(emotions)}")

        # Analyze entities for context
        if entities:
            if isinstance(entities[0], dict):
                entity_types = [e.get('type', 'unknown') for e in entities]
            else:
                entity_types = entities  # Assume list of strings
            meanings.append(f"Key entities: {', '.join(entity_types)}")

        # Look for patterns in past interactions
        if past_docs:
            past_emotions = []
            for doc in past_docs:
                try:
                    data = json.loads(doc)
                    past_emotions.extend(data.get('tone', {}).get('emotions', []))
                except:
                    continue
            if past_emotions:
                meanings.append(f"Pattern from history: {Counter(past_emotions).most_common(1)[0][0]}")

        return "; ".join(meanings) if meanings else "Surface level understanding"

    def _infer_true_intent(self, transcript, nlu_result, tone, past_docs):
        """
        Infers the user's true intent beyond spoken words.
        """
        # Simple intent inference based on patterns
        transcript_lower = transcript.lower()

        # Check for seeking help patterns
        help_keywords = ['help', 'advice', 'support', 'confused', 'lost', 'need']
        if any(word in transcript_lower for word in help_keywords):
            return "seeking_support"

        # Check for emotional expression
        if tone.get('overall_mood') == 'negative':
            return "expressing_distress"

        # Check for sharing experiences
        share_keywords = ['happened', 'experienced', 'went through', 'story']
        if any(word in transcript_lower for word in share_keywords):
            return "sharing_experience"

        return "general_conversation"

    def _calculate_understanding_confidence(self, nlu_result, tone):
        """
        Calculates confidence in the understanding.
        """
        # Simple confidence calculation
        nlu_confidence = 0.5  # Placeholder, assume average
        tone_confidence = tone.get('sentiment', {}).get('compound_score', 0)
        return (nlu_confidence + abs(tone_confidence)) / 2

    def detect_uncertainty_and_misunderstanding(self, understanding_result):
        """
        Detects uncertainty or potential misunderstanding in the analysis.
        """
        confidence = understanding_result.get('confidence', 0)
        deeper_meaning = understanding_result.get('deeper_meaning', '')

        uncertainty_indicators = []
        misunderstanding_risks = []

        if confidence < 0.6:
            uncertainty_indicators.append("Low confidence in analysis")

        if "Surface level" in deeper_meaning:
            misunderstanding_risks.append("May not have captured deeper intent")

        if not understanding_result.get('context_relevance', False):
            uncertainty_indicators.append("Limited historical context")

        return {
            "has_uncertainty": len(uncertainty_indicators) > 0,
            "has_misunderstanding_risk": len(misunderstanding_risks) > 0,
            "indicators": uncertainty_indicators + misunderstanding_risks
        }

    def generate_internal_response(self, understanding_result, insights, tone, uncertainty_analysis):
        """
        Generates an internal response based on deep understanding.
        """
        true_intent = understanding_result.get('true_intent', 'general_conversation')
        deeper_meaning = understanding_result.get('deeper_meaning', '')
        emotions = tone.get('emotions', [])

        # Base response based on intent
        if true_intent == "seeking_support":
            response = "I sense you're looking for support. "
        elif true_intent == "expressing_distress":
            response = "I can hear that you're going through a difficult time. "
        elif true_intent == "sharing_experience":
            response = "Thank you for sharing your experience with me. "
        else:
            response = "I understand what you're saying. "

        # Add emotional acknowledgment
        if emotions:
            response += f"I notice you're feeling {', '.join(emotions)}. "

        # Add insights if available
        if insights.get('past_connections'):
            response += "This connects to what you've shared before. "

        # Handle uncertainty
        if uncertainty_analysis.get('has_uncertainty'):
            response += "I'm working to understand this better. "

        # Add therapeutic element
        response += "How can I best support you right now?"

        return response

    def self_reflect_on_response(self, response, user_feedback, understanding_result):
        """
        Reflects on the generated response for improvement.
        """
        reflection = {
            "response_quality": "good",
            "areas_for_improvement": [],
            "learned_insights": []
        }

        # Analyze response length and empathy
        if len(response.split()) < 10:
            reflection["areas_for_improvement"].append("Response too brief")

        empathy_words = ['understand', 'sense', 'hear', 'feel', 'support']
        if not any(word in response.lower() for word in empathy_words):
            reflection["areas_for_improvement"].append("Could be more empathetic")

        # Reflect on understanding accuracy
        if understanding_result.get('confidence', 0) < 0.7:
            reflection["learned_insights"].append("Need better confidence thresholds")

        return reflection

    def learn_from_interaction(self, understanding_result, response, next_message=""):
        """
        Learns from the interaction to improve future responses.
        """
        learning_points = []

        # Store successful patterns
        if understanding_result.get('confidence', 0) > 0.8:
            learning_points.append("High confidence understanding achieved")

        # Learn from response effectiveness (placeholder)
        if next_message:
            learning_points.append("User continued conversation")

        # Store in long-term memory for future reference
        learning_data = {
            "understanding": understanding_result,
            "response": response,
            "learning_points": learning_points
        }
        self.ltm.store(json.dumps(learning_data))

        return learning_points

    def understand_user_meaning(self, transcript, nlu_result):
        """
        Understands what the user truly means, beyond spoken words.
        """
        # Use deep semantic understanding
        tone_placeholder = {"emotions": [], "sentiment": {"compound_score": 0}}
        understanding = self.deep_semantic_understanding(transcript, nlu_result, tone_placeholder)
        return understanding

    def identify_hidden_emotions(self, tone, nlu_result):
        """
        Identifies hidden or mixed emotions.
        """
        emotions = tone.get('emotions', [])
        sentiment = tone.get('sentiment', {})

        hidden_emotions = []

        # Check for mixed emotions (positive and negative)
        if 'happy' in emotions and any(e in emotions for e in ['sad', 'angry', 'fear']):
            hidden_emotions.append("mixed_emotions")

        # Check for suppressed emotions based on sentiment vs stated emotions
        polarity = sentiment.get('polarity', 0)
        if polarity < -0.5 and 'happy' in emotions:
            hidden_emotions.append("suppressed_negative")

        return {"hidden_emotions": hidden_emotions}

    def reflect_on_responses(self, previous_responses):
        """
        Reflects on its own responses.
        """
        if not previous_responses:
            return {"reflection": "No previous responses to reflect on"}

        # Analyze patterns in responses
        response_lengths = [len(r.split()) for r in previous_responses]
        avg_length = sum(response_lengths) / len(response_lengths)

        empathy_count = sum(1 for r in previous_responses if any(word in r.lower() for word in ['understand', 'feel', 'support']))

        reflection = {
            "average_response_length": avg_length,
            "empathy_frequency": empathy_count / len(previous_responses),
            "overall_quality": "good" if empathy_count > len(previous_responses) / 2 else "needs_improvement"
        }

        return {"reflection": reflection}

    def recognize_uncertainty(self, confidence_scores):
        """
        Recognizes uncertainty or misunderstanding.
        """
        # confidence_scores is a dict with various scores
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0

        return {"uncertainty": avg_confidence < 0.6}

    def learn_from_interactions(self, interaction_data):
        """
        Learns from past interactions to improve.
        """
        # interaction_data is a list of past interactions
        if not interaction_data:
            return {"learning": "No data to learn from"}

        # Simple learning: count successful patterns
        successful_interactions = sum(1 for data in interaction_data if data.get('success', False))

        learning = f"Learned from {successful_interactions} successful interactions out of {len(interaction_data)}"

        return {"learning": learning}
