from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_from_directory, session
import tempfile
import os
import json
import uuid
import assemblyai as aai
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from utils.llm_client import generate_chat_response, validate_groq_api_key
from gtts import gTTS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from perception.stt.stt_live import save_wav, transcribe_audio
import requests

# Stub function for analyze_tone
def analyze_tone(text):
    return {"sentiment": "neutral", "emotion": "neutral", "confidence": 0.5}

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
from core.ethics_personalization import EthicalAwarenessEngine, PersonalizationEngine

# Ensure static/audio exists
os.makedirs(os.path.join('static', 'audio'), exist_ok=True)


load_dotenv()
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

# Initialize Flask application
app = Flask(__name__)

# Security Config
flask_secret = os.environ.get('FLASK_SECRET_KEY')
if not flask_secret:
    print('Warning: FLASK_SECRET_KEY not set; using a generated (non-persistent) secret key.')
    flask_secret = os.urandom(24).hex()
app.secret_key = flask_secret

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
# FIX 1: Pylance sometimes flags this, but it works in Flask-Login. We keep it standard.
login_manager.login_view = 'login' # type: ignore

# --- DATABASE SETUP (Robust Fallback) ---
users = {} # In-memory fallback
mongo_connected = False
users_collection = None
db = None

try:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb+srv://abc:1234@cluster0.jlrvd9l.mongodb.net/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    
    db = client['agi-therapist']
    users_collection = db['users']
    mongo_connected = True
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    print("   -> Switching to IN-MEMORY storage (Changes will be lost on restart)")
    mongo_connected = False
    users_collection = None

# --- GLOBAL MODULES ---
memory_store = ServerMemoryStore()
prompt_builder = PromptBuilder(model="Meta-Llama-3.3-70B-Instruct")
wm = WorkingMemory()
agent = AGI119Agent()
safety_engine = EthicalAwarenessEngine()
style_engine = PersonalizationEngine()


# --- USER CLASS ---
class User(UserMixin):
    def __init__(self, user_id, name, email, password_hash):
        self.id = user_id
        self.name = name
        self.email = email
        self.password = password_hash

@login_manager.user_loader
def load_user(user_id):
    try:
        if mongo_connected and users_collection is not None:
            user_data = users_collection.find_one({"email": user_id})
            if user_data:
                return User(
                    user_id=user_data['email'],
                    name=user_data.get('name', user_data['email']),
                    email=user_data['email'],
                    password_hash=user_data['password']
                )
        else:
            # Fallback to in-memory
            # Note: We need to recreate the User object from the dictionary if needed
            u = users.get(user_id)
            if u:
                return u
    except Exception as e:
        print(f"Error loading user: {e}")
    return None

# --- AUDIO GENERATION ---
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

# --- GENERATION LOGIC ---
def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript="", conversation_id=None):
    try:
        # Safety check
        if safety_engine.detect_high_risk(transcript):
            print("⚠️ HIGH RISK DETECTED - Triggering Safety Protocol")
            response_text = safety_engine.ethical_response()
        else:
            # Retrieve memories
            retrieved_bundle = retrieve_memories(user_id, transcript)
            # Gather reasoning
            reasoning_data = gather_reasoning(user_id, tone, retrieved_bundle)
            # Build prompt
            prompt_data = build_prompt(user_id, transcript, retrieved_bundle, reasoning_data)
            # Call LLM
            response_text = call_llm(prompt_data)

        audio_path = generate_audio(response_text)
        stored_conversation_id = store_conversation(user_id, transcript, response_text, conversation_id)

        return {"text": response_text, "audio": audio_path, "conversation_id": stored_conversation_id}

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return {"text": "I'm listening. Please go on.", "audio": None, "conversation_id": conversation_id}

def retrieve_memories(user_id, transcript):
    try:
        return memory_store.retrieve_memories(user_id, transcript, top_k=5)
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        return {"profile_summary": "", "top_memories": [], "recency_window": [], "risk_flags": []}

def gather_reasoning(user_id, tone, retrieved_bundle):
    try:
        user_life = UserLifeUnderstanding(user_id)
        # Safe calls with fallbacks
        life_story = user_life.build_life_story() if hasattr(user_life, 'build_life_story') else {}
        emotional_progress = "Stable"
        recurring_problems = []
        
        # Check if methods exist before calling (defensive coding)
        if hasattr(user_life, 'recognize_emotional_progress'):
             emotional_progress = user_life.recognize_emotional_progress()
        if hasattr(user_life, 'analyze_recurring_problems'):
             probs = user_life.analyze_recurring_problems()
             if isinstance(probs, dict):
                 recurring_problems = probs.get('recurring_problems', [])

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
        # FIX: Updated fallback model to gemini-2.5-flash-lite
        return {
            "messages": [{"role": "user", "content": transcript}], 
            "model": "gemini-2.5-flash-lite"
        }

def call_llm(prompt_data):
    try:
        # 1. Get Key
        api_key = session.get('gemini_api_key') or os.environ.get("GEMINI_API_KEY")
        
        # 2. Extract Messages
        messages = prompt_data.get('messages') if isinstance(prompt_data, dict) else prompt_data
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]

        # 3. Call Client (Force the model name here too, just to be safe)
        return generate_chat_response(messages=messages, model="gemini-2.5-flash-lite", api_key=api_key)

    except Exception as e:
        print(f"LLM error: {str(e)}")
        return "I understand. Please go on."

def store_conversation(user_id, transcript, response_text, conversation_id=None):
    try:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # Store user message
        memory_store.store_memory(user_id, "conversation", f"User: {transcript}", 
                                tags=["conversation", "user_message", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id)
        
        # Store AI response
        memory_store.store_memory(user_id, "conversation", f"AI: {response_text}", 
                                tags=["conversation", "ai_message", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id)
        return conversation_id
    except Exception as e:
        print(f"Error storing conversation: {str(e)}")
        return None


# --- ROUTES ---

@app.route('/')
@login_required
def index():
    user_id = current_user.id
    user_settings = get_user_settings(user_id) or get_default_settings()
    if not isinstance(user_settings, dict):
        user_settings = get_default_settings()
    
    return render_template('index.html', 
                         user_name=current_user.name,
                         user_email=current_user.email,
                         user_settings=user_settings)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        try:
            user_data = None
            if mongo_connected and users_collection is not None:
                user_data = users_collection.find_one({"email": email})
            else:
                user_data = users.get(email) # Check in-memory
                # Since 'users' stores User objects, we need to extract dict data if simulating DB
                if user_data: 
                     user_data = {'email': user_data.email, 'name': user_data.name, 'password': user_data.password}
                
            if not user_data:
                flash('Account not found')
                return render_template('login.html')
                
            if check_password_hash(user_data['password'], password):
                user = User(
                    user_id=user_data['email'],
                    name=user_data.get('name', user_data['email']),
                    email=user_data['email'],
                    password_hash=user_data['password']
                )
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Invalid password')
        except Exception as e:
            print(f"Login error: {e}")
            flash('Login service unavailable - try again later')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            if not name or not email or not password:
                flash('All fields are required')
                return render_template('signup.html')
                
            # Check existing
            existing_user = None
            if mongo_connected and users_collection is not None:
                existing_user = users_collection.find_one({"email": email})
            else:
                existing_user = users.get(email)
                
            if existing_user:
                flash('Email already exists')
                return render_template('signup.html')
                
            # Create user
            password_hash = generate_password_hash(password)
            
            if mongo_connected and users_collection is not None:
                users_collection.insert_one({
                    "email": email,
                    "name": name,
                    "password": password_hash
                })
            else:
                # Store in memory
                new_user_obj = User(email, name, email, password_hash)
                users[email] = new_user_obj
                print(f"📝 User {name} registered in MEMORY (Temporary)")
                
            # Auto login
            new_user = User(email, name, email, password_hash)
            login_user(new_user)
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Signup error: {e}")
            flash('Error creating account - please try again')
            
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_id = current_user.id
    user_settings = get_user_settings(user_id) or get_default_settings()
    return render_template('settings.html', current_key='', user_settings=user_settings)

# --- SETTINGS HELPERS ---
def get_user_settings(user_id):
    if mongo_connected and users_collection is not None:
        try:
            user_doc = users_collection.find_one({"email": user_id})
            if user_doc and 'settings' in user_doc:
                return user_doc['settings']
        except Exception as e:
            print(f"Error getting user settings: {str(e)}")
    return get_default_settings()

def get_default_settings():
    return {
        'dark_mode': False,
        'max_tokens': 500,
        'therapeutic_style': 'empathetic',
        'enable_audio': True,
        'theme_color': '#667eea'
    }

# --- API ROUTES ---
@app.route('/start_conversation', methods=['POST'])
@login_required
def start_conversation():
    try:
        conversation_id = request.form.get('conversation_id') or str(uuid.uuid4())
        greeting = "Hello! I'm your AI therapist. How are you feeling today?"
        return jsonify({"message": greeting, "type": "bot", "conversation_id": conversation_id})
    except Exception as e:
        print(f"Error in start_conversation: {str(e)}")
        return jsonify({"error": "Error starting conversation"}), 500

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        user_id = current_user.id
        conversation_id = request.form.get('conversation_id')
        
        transcript = get_transcript_from_request()
        if not transcript:
            return jsonify({"error": "No text or audio provided"}), 400
        
        perception_data = analyze_perception(transcript)
        response_data = generate_response_data(perception_data, user_id, transcript, conversation_id)
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        return jsonify({"error": "Analysis failed"}), 500

def get_transcript_from_request():
    if 'text' in request.form and request.form['text'].strip():
        return request.form['text'].strip()
    return None

def analyze_perception(transcript):
    try:
        tone = analyze_tone(transcript)
        nlu_result = nlu_process(transcript, tone)
        return {"transcript": transcript, "tone": tone, "nlu": nlu_result}
    except Exception:
        return {"transcript": transcript, "tone": {"overall_mood": "neutral"}, "nlu": {}}

def generate_response_data(perception_data, user_id, transcript, conversation_id=None):
    insights = {}
    # This calls the main logic to get text and audio
    response = generate_therapist_response(perception_data, insights, perception_data['tone'], user_id, transcript, conversation_id)
    
    response_data = {
        "message": response.get("text", "I understand."),
        "type": "bot",
        "conversation_id": response.get("conversation_id", conversation_id),
        "analysis": perception_data
    }
    
    # FIX: Adding '/static' so the browser can find the file
    if response.get("audio"):
        filename = os.path.basename(response['audio'])
        response_data["audio_url"] = f"/static/audio/{filename}"
        
    return response_data

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)