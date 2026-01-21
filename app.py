from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_from_directory
import tempfile
import os
import json
import assemblyai as aai
import openai
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from gtts import gTTS
import pyttsx3
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory
from reasoning.user_life_understanding import UserLifeUnderstanding
from reasoning.emotional_reasoning import EmotionalReasoning
from reasoning.ethical_awareness import EthicalAwareness
from reasoning.internal_cognition import InternalCognition
from perception.reasoning.insight import InsightGenerator
from core.agi_agent import AGI119Agent
from core.emotion_detector import detect_emotion
from api.memory_store import ServerMemoryStore
from prompt_builder.prompt_builder import PromptBuilder

# --- NEW INTEGRATION: Teammate's Safety Module ---
# We use the class directly to avoid conflicts
from core.ethics_personalization import EthicalAwarenessEngine, PersonalizationEngine

load_dotenv()

aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "4bedc386183f491b9d12365c4d91e1a3")
# SambaNova setup
openai.api_base = "https://api.sambanova.ai/v1/"
config_file = 'config.json'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        openai.api_key = config.get('api_key', os.environ.get("SAMBA_API_KEY", "587a7fba-09f4-4bb5-a0bf-7a359629d44b"))
else:
    openai.api_key = os.environ.get("SAMBA_API_KEY", "587a7fba-09f4-4bb5-a0bf-7a359629d44b")

# MongoDB setup
mongo_uri = "mongodb+srv://abc:1234@cluster0.jlrvd9l.mongodb.net/"
client = MongoClient(mongo_uri)
db = client['agi-therapist']  # Database name
users_collection = db['users']

# Initialize global modules first
memory_store = ServerMemoryStore()
prompt_builder = PromptBuilder(model="Meta-Llama-3.3-70B-Instruct")

# Initialize Global Modules
wm = WorkingMemory()
wm_logs = []
ltm_logs = []
agent = AGI119Agent()

# Initialize Safety Engines (Teammate's Code)
safety_engine = EthicalAwarenessEngine()
style_engine = PersonalizationEngine()

def generate_audio(text):
    try:
        import uuid
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join('static', 'audio', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript=""):
    try:
        # Safety check
        if safety_engine.detect_high_risk(transcript):
            print("⚠️ HIGH RISK DETECTED - Triggering Safety Protocol")
            return {"text": safety_engine.ethical_response(), "audio": None}

        # Retrieve memories
        retrieved_bundle = retrieve_memories(user_id, transcript)

        # Gather reasoning
        reasoning_data = gather_reasoning(user_id, tone, retrieved_bundle)

        # Build prompt
        prompt_data = build_prompt(user_id, transcript, retrieved_bundle, reasoning_data)

        # Call LLM
        response_text = call_llm(prompt_data)

        # Generate audio
        audio_path = generate_audio(response_text)

        # Store conversation
        store_conversation(user_id, transcript, response_text)

        print(f"Prompt tokens: {prompt_data['token_count']}")
        print(f"Response: {response_text}")

        return {"text": response_text, "audio": audio_path}

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return {"text": "I'm listening. Please go on.", "audio": None}

def retrieve_memories(user_id, transcript):
    try:
        return memory_store.retrieve_memories(user_id, transcript, top_k=5)
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        return {"profile_summary": "", "top_memories": [], "recency_window": [], "risk_flags": []}

def gather_reasoning(user_id, tone, retrieved_bundle):
    try:
        user_life = UserLifeUnderstanding(user_id)
        life_story = user_life.build_lifeA_story()
        emotional_progress = user_life.recognize_emotional_progress()
        recurring_problems = user_life.analyze_recurring_problems()['recurring_problems']

        emotional_reasoning = EmotionalReasoning()
        history = [mem['text'] for mem in retrieved_bundle.get('top_memories', [])]
        therapeutic_insight = emotional_reasoning.provide_therapeutic_insight(tone.get('emotions', []), history)

        return {
            'life_story': life_story,
            'emotional_progress': emotional_progress,
            'recurring_problems': recurring_problems,
            'therapeutic_insight': therapeutic_insight
        }
    except Exception as e:
        print(f"Error gathering reasoning: {str(e)}")
        return {}

def build_prompt(user_id, transcript, retrieved_bundle, reasoning_data):
    try:
        style_config = {"style": "medium", "therapeutic": True}
        return prompt_builder.build_prompt(user_id, transcript, retrieved_bundle, style_config, reasoning_data)
    except Exception as e:
        print(f"Error building prompt: {str(e)}")
        return {"messages": [{"role": "user", "content": transcript}], "debug_prompt_text": transcript, "token_count": len(transcript)}

def call_llm(prompt_data):
    try:
        response = openai.ChatCompletion.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=prompt_data["messages"],
            max_tokens=500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"LLM error: {str(e)}")
        # Fallback mock response for demo
        return "I understand you're going through a difficult time. Can you tell me more about what's on your mind?"

def store_conversation(user_id, transcript, response_text):
    try:
        memory_store.store_memory(user_id, "conversation", f"User: {transcript}\nAI: {response_text}", tags=["conversation"])
    except Exception as e:
        print(f"Error storing conversation: {str(e)}")

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, name, email, password):
        self.id = id
        self.name = name
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    user_data = users.get(user_id)
    if user_data:
        return user_data
    return None

def load_users():
    try:
        users_data = users_collection.find()
        return {user['email']: User(user['email'], user.get('name', user.get('email', '')), user.get('email', user.get('email', '')), user['password']) for user in users_data}
    except Exception as e:
        print(f"Error loading users from MongoDB: {e}")
        return {}

def save_users():
    try:
        # Clear existing users
        users_collection.delete_many({})
        # Insert current users
        users_data = [{'email': k, 'name': v.name, 'password': v.password} for k, v in users.items()]
        if users_data:
            users_collection.insert_many(users_data)
    except Exception as e:
        print(f"Error saving users to MongoDB: {e}")

users = load_users()

# Define the route for the index page
@app.route('/')
@login_required
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        if email in users:
            flash('Email already exists')
        else:
            user = User(email, name, email, password)
            users[email] = user
            save_users()
            login_user(user)
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    config_file = 'config.json'
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        if api_key:
            config = {}
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            config['api_key'] = api_key
            with open(config_file, 'w') as f:
                json.dump(config, f)
            openai.api_key = api_key
            flash('API Key updated successfully')
        else:
            flash('Please provide an API Key')
    # Load current api key to display (masked)
    current_key = ''
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            current_key = config.get('api_key', '')
            if current_key:
                current_key = '*' * (len(current_key) - 4) + current_key[-4:]  # Mask most of it
    return render_template('settings.html', current_key=current_key)

# Define the route for starting a conversation
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    try:
        user_id = request.form.get('user_id', 'default')

        # Simplified for testing
        greeting = "Hello! I'm your AI therapist. How are you feeling today? You can type your message or record audio."

        return jsonify({"message": greeting, "type": "bot"})
    except Exception as e:
        print(f"Error in start_conversation: {str(e)}")
        return jsonify({"error": "An error occurred starting the conversation. Please try again."}), 500

# Define the route for analyzing audio input
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        user_id = request.form.get('user_id', 'default')
        
        # Get transcript from text or audio
        transcript = get_transcript_from_request()
        if not transcript:
            return jsonify({"error": "No text or audio provided"}), 400
        
        # Analyze perception
        perception_data = analyze_perception(transcript)
        
        # Generate response
        response_data = generate_response_data(perception_data, user_id, transcript)
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        return jsonify({"error": "An error occurred during analysis. Please try again."}), 500

def get_transcript_from_request():
    if 'text' in request.form and request.form['text'].strip():
        return request.form['text'].strip()
    elif 'audio' in request.files:
        try:
            audio_file = request.files['audio']
            temp_path = tempfile.mktemp(suffix='.wav')
            audio_file.save(temp_path)
            transcript = transcribe_audio(temp_path)
            os.unlink(temp_path)
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None
    return None

def analyze_perception(transcript):
    try:
        tone = analyze_tone(transcript)
        nlu_result = nlu_process(transcript, tone)
        return {"transcript": transcript, "tone": tone, "nlu": nlu_result}
    except Exception as e:
        print(f"Error in perception analysis: {str(e)}")
        return {"transcript": transcript, "tone": {"overall_mood": "neutral"}, "nlu": {"entities": [], "semantic_roles": []}}

def generate_response_data(perception_data, user_id, transcript):
    # Generate insights (placeholder)
    insights = {}
    
    # Generate response
    response = generate_therapist_response(perception_data, insights, perception_data['tone'], user_id, transcript)
    
    response_data = {
        "message": response.get("text", "I understand. Tell me more."),
        "type": "bot",
        "analysis": perception_data
    }
    if 'audio' in request.files:
        response_data["transcript"] = transcript
    if response.get("audio"):
        response_data["audio_url"] = f"/audio/{os.path.basename(response['audio'])}"
    
    return response_data

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
    try:
        user_id = request.args.get('user_id', 'default')
        ltm = LongTermMemory(user_id=user_id)
        ltm.store("test", "1")
        result = ltm.retrieve("test")
        ltm.update("1", "test2")
        result2 = ltm.retrieve("test2")
        return jsonify({"result": result, "result2": result2})
    except Exception as e:
        return jsonify({"error": f"LTM test failed: {str(e)}"}), 500

@app.route('/test')
def test():
    return jsonify({"status": "ok"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
