from flask import Flask, render_template, jsonify, request
import tempfile
import os
import json
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory
from reasoning.user_life_understanding import UserLifeUnderstanding
from reasoning.internal_cognition import InternalCognition
# New Import for the Insight module
from perception.reasoning.insight import InsightGenerator 
from config import validate_api_keys

# Validate API keys on startup
api_errors = validate_api_keys()
if api_errors:
    print("❌ API Configuration Errors:")
    for error in api_errors:
        print(f"  - {error}")
    print("\nPlease configure the missing API keys in config.py before running the application.")
    exit(1)

app = Flask(__name__)

# Initialize memory modules
wm = WorkingMemory()
wm_logs = [] 
ltm_logs = [] 

# --- CORE FUNCTIONS ---

def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript=""):
    try:
        # Initialize Internal Cognition system
        cognition = InternalCognition(user_id=user_id)

        # Perform deep semantic understanding
        understanding_result = cognition.deep_semantic_understanding(transcript, perception_result, tone)

        # Detect uncertainty and potential misunderstandings
        uncertainty_analysis = cognition.detect_uncertainty_and_misunderstanding(understanding_result)

        # Generate response using internal cognition
        response_text = cognition.generate_internal_response(
            understanding_result,
            insights,
            tone,
            uncertainty_analysis
        )

        # Self-reflection (logging only)
        reflection = cognition.self_reflect_on_response(
            response_text,
            "", 
            understanding_result
        )

        # Log the internal cognition process
        cognition_log = {
            'understanding': understanding_result,
            'uncertainty': uncertainty_analysis,
            'reflection': reflection,
            'response': response_text
        }
        print(f"Internal Cognition Analysis: {json.dumps(cognition_log, indent=2)}")

        return response_text

    except Exception as e:
        print(f"Unexpected error in generate_therapist_response: {str(e)}")
        return generate_fallback_response(tone, insights)

def generate_fallback_response(tone, insights):
    """Generate a therapeutic response when internal cognition fails."""
    try:
        sentiment = tone.get('sentiment', 'neutral') if tone else 'neutral'
        confidence = tone.get('confidence', 0.5) if tone else 0.5

        if sentiment == 'negative' and confidence > 0.6:
            response = "I can sense this is difficult for you right now. It's completely valid to feel this way. "
        elif sentiment == 'positive' and confidence > 0.6:
            response = "I'm glad to hear you're feeling positive about this. "
        else:
            response = "Thank you for sharing that with me. "

        # Add insights if available
        if insights:
            if insights.get('past_connections'):
                response += "I notice this connects to what you've shared before. "
            if insights.get('cognitive_insights') and insights['cognitive_insights'].get('needs_intervention'):
                 response += "It seems like we've hit a recurring theme here. "

        follow_ups = [
            "How can I best support you in this moment?",
            "What would be most helpful for you to explore right now?",
            "How are you feeling about sharing this with me?"
        ]
        import random
        response += random.choice(follow_ups)

        return response
    except Exception as e:
        return "Thank you for sharing that. I'm here to listen. How are you feeling right now?"

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    user_id = request.form.get('user_id', 'default')
    ltm = LongTermMemory(user_id=user_id)
    try:
        previous_data = ltm.get_all()
        # Safe check for previous conversations
        documents = previous_data.get('documents') or []
        if documents and len(documents) > 0:
            greeting = "Welcome back! I remember we've talked before. How are you feeling today?"
        else:
            greetings = [
                "Hello! I'm your AI therapist. I'm here to listen without judgment. How are you doing today?",
                "Hi there! Welcome. I'm here to support you. What's on your mind?"
            ]
            import random
            greeting = random.choice(greetings)
    except:
        greeting = "Hello! I'm your AI therapist. How are you feeling today?"

    return jsonify({"message": greeting, "type": "bot"})

@app.route('/analyze', methods=['POST'])
def analyze():
    print("\n--- NEW REQUEST ---")
    try:
        user_id = request.form.get('user_id', 'default')
        ltm = LongTermMemory(user_id=user_id)

        # 1. Get Input
        if 'text' in request.form and request.form['text'].strip():
            transcript = request.form['text'].strip()
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                filename = f.name
            audio_file.save(filename)
            transcript = transcribe_audio(filename)
            os.unlink(filename)
        else:
            return jsonify({"error": "No text or audio provided"}), 400

        # Fix transcript if it's a list (common bug)
        if isinstance(transcript, list):
            transcript = " ".join(transcript)

        # 2. Perception
        tone = analyze_tone(transcript)
        result = nlu_process(transcript, tone)

        # 3. Reasoning & Insights
        ulu = UserLifeUnderstanding(user_id=user_id)
        
        # Initialize Insight Generator
        ig = InsightGenerator(user_id=user_id)
        additional_insights = ig.generate(transcript)

        insights = {
            'past_connections': ulu.connect_past_present(transcript),
            'recurring_problems': ulu.analyze_recurring_problems(),
            'life_story': ulu.build_life_story(),
            'emotional_progress': ulu.recognize_emotional_progress(),
            'consistency_context': ulu.maintain_consistency(transcript),
            'cognitive_insights': additional_insights
        }

        # 4. Memory Storage
        wm.store(json.dumps(result), str(len(wm_logs or [])))
        wm_logs.append(result)
        ltm.store(json.dumps(result), str(len(ltm_logs or [])))
        ltm_logs.append(result)

        # 5. Response Generation
        response_text = generate_therapist_response(result, insights, tone, user_id, transcript)

        return jsonify({
            "message": response_text,
            "type": "bot",
            "analysis": {
                "perception": result,
                "tone": tone,
                "insights": insights
            }
        })

    except Exception as e:
        print(f"CRITICAL ERROR in /analyze: {str(e)}")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)