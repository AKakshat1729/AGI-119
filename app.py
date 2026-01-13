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
from perception.reasoning.insight import InsightGenerator 
from config import validate_api_keys

# --- NEW INTEGRATION: Teammate's Safety Module ---
# We use the class directly to avoid conflicts
from core.ethics_personalization import EthicalAwarenessEngine, PersonalizationEngine

# Validate API keys
api_errors = validate_api_keys()
if api_errors:
    print("❌ API Configuration Errors:")
    for error in api_errors:
        print(f"  - {error}")
    exit(1)

app = Flask(__name__)

# Initialize Global Modules
wm = WorkingMemory()
wm_logs = [] 
ltm_logs = [] 

# Initialize Safety Engines (Teammate's Code)
safety_engine = EthicalAwarenessEngine()
style_engine = PersonalizationEngine()

# --- CORE FUNCTIONS ---

def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript=""):
    try:
        # 1. SAFETY CHECK (Priority #1)
        if safety_engine.detect_high_risk(transcript):
            print("⚠️ HIGH RISK DETECTED - Triggering Safety Protocol")
            return safety_engine.ethical_response()

        # 2. Normal Processing
        cognition = InternalCognition(user_id=user_id)
        
        # Determine basic emotion label for Style Engine
        sentiment_score = tone.get('sentiment_score', 0)
        simple_emotion = "neutral"
        if sentiment_score > 0.2: simple_emotion = "happy"
        elif sentiment_score < -0.2: simple_emotion = "sad"
        
        # Get suggested tone style
        suggested_style = style_engine.adapt_tone(simple_emotion)
        print(f"🎨 Style Engine suggests: {suggested_style}")

        # 3. Deep Understanding
        understanding_result = cognition.deep_semantic_understanding(transcript, perception_result, tone)
        
        # Detect uncertainty
        uncertainty_analysis = cognition.detect_uncertainty_and_misunderstanding(understanding_result)

        # 4. Generate Response
        response_text = cognition.generate_internal_response(
            understanding_result,
            insights,
            tone,
            uncertainty_analysis
        )

        # --- RESTORED: Self Reflection & Logging ---
        reflection = cognition.self_reflect_on_response(
            response_text,
            "", 
            understanding_result
        )

        cognition_log = {
            'understanding': understanding_result,
            'uncertainty': uncertainty_analysis,
            'reflection': reflection,
            'style_suggested': suggested_style,  # Added this so you can see the new feature too!
            'response': response_text
        }
        print(f"Internal Cognition Analysis: {json.dumps(cognition_log, indent=2)}")
        # -------------------------------------------

        return response_text

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return "I'm listening. Please go on."

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    user_id = request.form.get('user_id', 'default')
    # Simple greeting logic
    return jsonify({"message": "Hello! I'm your AI therapist. I'm here to listen. How are you?", "type": "bot"})

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
                return jsonify({"error": "No file"}), 400
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                filename = f.name
            audio_file.save(filename)
            transcript = transcribe_audio(filename)
            os.unlink(filename)
        else:
            return jsonify({"error": "No input"}), 400

        if isinstance(transcript, list): transcript = " ".join(transcript)

        # 2. Perception Pipeline
        tone = analyze_tone(transcript)
        result = nlu_process(transcript, tone)

        # 3. Reasoning & Insights
        ulu = UserLifeUnderstanding(user_id=user_id)
        ig = InsightGenerator(user_id=user_id)
        
        additional_insights = ig.generate(transcript)
        insights = {
            'past_connections': ulu.connect_past_present(transcript),
            'life_story': ulu.build_life_story(),
            'cognitive_insights': additional_insights
        }

        # 4. Memory Storage
        wm.store(json.dumps(result), str(len(wm_logs)))
        wm_logs.append(result)
        ltm.store(json.dumps(result), str(len(ltm_logs)))
        ltm_logs.append(result)

        # 5. Response
        response_text = generate_therapist_response(result, insights, tone, user_id, transcript)

        return jsonify({
            "message": response_text,
            "type": "bot",
            "analysis": {"perception": result, "insights": insights}
        })

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)