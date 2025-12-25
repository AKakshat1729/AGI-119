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

# Function to generate therapist-like responses
def generate_therapist_response(perception_result, insights, tone):
    # Base responses based on sentiment
    sentiment_responses = {
        'positive': [
            "I'm glad to hear you're feeling positive about this!",
            "That sounds encouraging. Tell me more about what brings you joy.",
            "It's wonderful that you're experiencing positive emotions."
        ],
        'negative': [
            "I can sense this is difficult for you right now.",
            "It sounds like you're going through a challenging time.",
            "I'm here to listen and support you through this."
        ],
        'neutral': [
            "Thank you for sharing that with me.",
            "I appreciate you opening up about this.",
            "Let's explore this together."
        ]
    }

    sentiment = tone.get('sentiment', 'neutral')
    base_response = sentiment_responses.get(sentiment, sentiment_responses['neutral'])[0]

    # Add emotional validation
    if sentiment == 'negative':
        base_response += " It's completely valid to feel this way. "
    elif sentiment == 'positive':
        base_response += " It's important to acknowledge these positive feelings. "

    # Add insights from reasoning
    if insights.get('past_connections'):
        connections = str(insights['past_connections'])[:150]
        if connections:
            base_response += f"I notice this connects to what you've shared before about {connections}... "

    if insights.get('emotional_progress'):
        progress = str(insights['emotional_progress'])[:100]
        if progress:
            base_response += f"I see some emotional growth here: {progress}. "

    if insights.get('recurring_problems'):
        patterns = str(insights['recurring_problems'])[:100]
        if patterns:
            base_response += f"We might be seeing some patterns emerge around {patterns}. "

    # Add follow-up questions
    follow_ups = [
        "How does this make you feel right now?",
        "What would you like to explore more about this?",
        "How can I best support you in this moment?",
        "What thoughts come up when you reflect on this?",
        "Is there anything specific you'd like to work on together?"
    ]

    import random
    follow_up = random.choice(follow_ups)
    base_response += follow_up

    return base_response

# Initialize Flask application
app = Flask(__name__)

# Initialize memory modules for short-term and long-term memory
wm = WorkingMemory() # Working memory instance

# In-memory logs for display in the web interface
wm_logs = [] # Logs for working memory operations
ltm_logs = [] # Logs for long-term memory operations

# Define the route for the index page
@app.route('/')
def index():
    # Render the index.html template
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
        # Get user_id from the request form data, default to 'default' if not provided
        user_id = request.form.get('user_id', 'default')

        # Initialize long-term memory instance for the specific user
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
        # Process the transcribed text and tone using natural language understanding
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
        response_text = generate_therapist_response(result, insights, tone)

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
        # Handle any exceptions and return an error message as a JSON response
        return jsonify({"error": str(e)}), 500

# Define a route to test the working memory
@app.route('/test_wm', methods=['GET'])
def test_wm():
    """
    Test the working memory by storing and retrieving a value.
    """
    wm.store({"test": "test"}, "1")
    result = wm.retrieve("test")
    wm.clear()
    return jsonify({"result": result})

# Define a route to test the long-term memory
@app.route('/test_ltm', methods=['GET'])
def test_ltm():
    """
    Test the long-term memory by storing, retrieving, and updating a value.
    """
    user_id = request.args.get('user_id', 'default')
    ltm = LongTermMemory(user_id=user_id)
    ltm.store("test", "1")
    result = ltm.retrieve("test")
    ltm.update("1", "test2")
    result2 = ltm.retrieve("test2")
    return jsonify({"result": result, "result2": result2})

if __name__ == '__main__':
    app.run(debug=True)
