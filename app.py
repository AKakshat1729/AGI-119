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
from config import validate_api_keys

# Validate API keys on startup
api_errors = validate_api_keys()
if api_errors:
    print("❌ API Configuration Errors:")
    for error in api_errors:
        print(f"  - {error}")
    print("\nPlease configure the missing API keys in config.py before running the application.")
    exit(1)

# Function to generate therapist-like responses using Internal Cognition and Self-Reflection
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

        # Perform self-reflection on the generated response
        # (This would typically happen after getting user feedback, but we include it for completeness)
        reflection = cognition.self_reflect_on_response(
            response_text,
            "",  # No immediate feedback available
            understanding_result
        )

        # Log the internal cognition process for analysis
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
        # Fallback to basic response
        return generate_fallback_response(tone, insights)

# Fallback response generator for when internal cognition fails
def generate_fallback_response(tone, insights):
    """Generate a therapeutic response when internal cognition fails."""
    try:
        sentiment = tone.get('sentiment', 'neutral') if tone else 'neutral'
        confidence = tone.get('confidence', 0.5) if tone else 0.5

        # Base response based on sentiment
        if sentiment == 'negative' and confidence > 0.6:
            response = "I can sense this is difficult for you right now. It's completely valid to feel this way, and it's brave of you to share it. "
        elif sentiment == 'positive' and confidence > 0.6:
            response = "I'm glad to hear you're feeling positive about this. It's important to acknowledge and celebrate these positive feelings. "
        else:
            response = "Thank you for sharing that with me. I appreciate you opening up about this. "

        # Add insights from user life understanding if available
        if insights:
            if insights.get('past_connections'):
                response += "I notice this connects to what you've shared before about your experiences. "
            if insights.get('recurring_problems'):
                response += "This seems to be something you've been working through. "
            if insights.get('emotional_progress'):
                response += "It's good to see how you're growing through these experiences. "

        # Add therapeutic follow-up
        follow_ups = [
            "How can I best support you in this moment?",
            "What would be most helpful for you to explore right now?",
            "How are you feeling about sharing this with me?",
            "What do you need most in this conversation?",
            "How has this been affecting your daily life?"
        ]

        import random
        response += random.choice(follow_ups)

        return response

    except Exception as e:
        print(f"Error in fallback response generation: {str(e)}")
        return "Thank you for sharing that with me. I'm here to listen and support you. How are you feeling right now?"

app = Flask(__name__)

# Initialize memory modules
wm = WorkingMemory()
wm_logs = [] 
ltm_logs = [] 

@app.route('/')
def index():
    return render_template('index.html')

# Define the route for starting a conversation
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    user_id = request.form.get('user_id', 'default')

    # Check if user has previous conversations
    ltm = LongTermMemory(user_id=user_id)
    try:
        previous_data = ltm.get_all()
        if previous_data and previous_data.get('documents') and len(previous_data['documents']) > 0:
            greeting = "Welcome back! I remember we've talked before. How are you feeling today? I'm here to listen and support you."
        else:
            greetings = [
                "Hello! I'm your AI therapist. I'm here to listen without judgment and help you explore your thoughts and feelings. How are you doing today?",
                "Hi there! Welcome to our conversation. I'm here to support you on your journey. What's on your mind today?",
                "Greetings! I'm glad you've reached out. Therapy is about creating a safe space for you to express yourself. How are you feeling right now?"
            ]
            import random
            greeting = random.choice(greetings)
    except:
        greeting = "Hello! I'm your AI therapist. How are you feeling today? You can type your message or record audio."

    return jsonify({"message": greeting, "type": "bot"})

# Define the route for analyzing audio input
@app.route('/analyze', methods=['POST'])
def analyze():
    print("Analyze route called")
    try:
        user_id = request.form.get('user_id', 'default')
        ltm = LongTermMemory(user_id=user_id)

        # Check if text is provided
        if 'text' in request.form and request.form['text'].strip():
            transcript = request.form['text'].strip()
        # Check if audio file is present in the request
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400

            # Save uploaded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                filename = f.name
            audio_file.save(filename)

            # Transcribe the audio file using speech-to-text
            transcript = transcribe_audio(filename)
            # Clean up the temporary audio file
            os.unlink(filename)
        else:
            return jsonify({"error": "No text or audio provided"}), 400
        # Analyze the tone of the transcribed text
        tone = analyze_tone(transcript)
        result = nlu_process(transcript, tone)

        # Initialize user life understanding
        ulu = UserLifeUnderstanding(user_id=user_id)

        # Get insights from long-term memory
        insights = {
            'past_connections': ulu.connect_past_present(transcript),
            'recurring_problems': ulu.analyze_recurring_problems(),
            'life_story': ulu.build_life_story(),
            'emotional_progress': ulu.recognize_emotional_progress(),
            'consistency_context': ulu.maintain_consistency(transcript)
        }

        # Store the result in working memory
        try:
            wm.store(json.dumps(result), str(len(wm_logs)))
            wm_logs.append(result)
        except Exception as e:
            wm_logs.append({"error": f"WM store failed: {str(e)}"})

        # Store the result in long-term memory
        try:
            ltm.store(json.dumps(result), str(len(ltm_logs)))
            ltm_logs.append(result)
        except Exception as e:
            ltm_logs.append({"error": f"LTM store failed: {str(e)}"})

        # Generate conversational response
        response_text = generate_therapist_response(result, insights, tone, user_id, transcript)

        # Learn from this interaction for future improvements
        try:
            cognition = InternalCognition(user_id=user_id)
            # Get the understanding result from the response generation (this is a simplified approach)
            understanding_result = cognition.deep_semantic_understanding(transcript, result, tone)
            # Learn from the interaction (next_message would be available in future interactions)
            learning_points = cognition.learn_from_interaction(
                understanding_result,
                response_text,
                ""  # Next message not available yet
            )
            print(f"Learning from interaction: {learning_points}")
        except Exception as learning_error:
            print(f"Learning error: {str(learning_error)}")

        # Return the conversational response
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)