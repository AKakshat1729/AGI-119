#!/usr/bin/env python3
"""
Test script for the Internal Cognition system
"""

from reasoning.internal_cognition import InternalCognition
import json

def test_internal_cognition():
    """Test the internal cognition system with a sample input."""

    # Sample inputs
    transcript = "I'm feeling really anxious about my job lately. I keep worrying that I'll make mistakes."
    nlu_result = {
        'intent': 'seeking_support',
        'confidence': 0.85,
        'entities': ['anxiety', 'job', 'worries']
    }
    tone_analysis = {
        'sentiment': 'negative',
        'confidence': 0.78,
        'emotions': ['anxiety', 'worry']
    }

    # Initialize cognition system
    cognition = InternalCognition(user_id="test_user")

    # Test deep semantic understanding
    print("=== Testing Deep Semantic Understanding ===")
    understanding = cognition.deep_semantic_understanding(transcript, nlu_result, tone_analysis)
    print(json.dumps(understanding, indent=2))

    # Test uncertainty detection
    print("\n=== Testing Uncertainty Detection ===")
    uncertainty = cognition.detect_uncertainty_and_misunderstanding(understanding)
    print(json.dumps(uncertainty, indent=2))

    # Test response generation
    print("\n=== Testing Response Generation ===")
    insights = {
        'past_connections': True,
        'recurring_problems': False,
        'emotional_progress': True
    }

    response = cognition.generate_internal_response(
        understanding,
        insights,
        tone_analysis,
        uncertainty
    )
    print(f"Generated Response: {response}")

    # Test self-reflection
    print("\n=== Testing Self-Reflection ===")
    reflection = cognition.self_reflect_on_response(response, "", understanding)
    print(json.dumps(reflection, indent=2))

    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_internal_cognition()