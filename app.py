from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_from_directory, session
import tempfile
import os
import json
import uuid
import warnings
import logging

# --- Suppress noisy startup warnings ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Suppress BertModel position_ids and HF hub warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from datetime import datetime
import assemblyai as aai
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from utils.llm_client import generate_chat_response, validate_gemini_api_key
from gtts import gTTS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
import requests
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
from reasoning.long_term_personalized_memory import PersonalizedMemoryModule

# --- NEW INTEGRATION: Teammate's Safety Module ---
from core.ethics_personalization import EthicalAwarenessEngine, PersonalizationEngine

# --- NEW: Lightweight Clinical Intelligence Layer ---
from core.clinical_intelligence import get_clinical_engine
clinical_engine = get_clinical_engine()

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
# --- USER CLASS & LOCAL STORAGE ---
USER_STORAGE_FILE = 'users.json'

class User(UserMixin):
    def __init__(self, user_id, name, email, password_hash, settings=None):
        self.id = user_id
        self.name = name
        self.email = email
        self.password = password_hash
        self.settings = settings or {}

    def to_dict(self):
        return {
            "email": self.email,
            "name": self.name,
            "password": self.password,
            "settings": self.settings
        }

def save_local_users(users_dict):
    try:
        data = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in users_dict.items()}
        with open(USER_STORAGE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving local users: {e}")

def load_local_users():
    if not os.path.exists(USER_STORAGE_FILE):
        return {}
    try:
        with open(USER_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # Reconstruct User objects
            return {k: User(v['email'], v['name'], v['email'], v['password'], v.get('settings')) for k, v in data.items()}
    except Exception as e:
        print(f"Error loading local users: {e}")
        return {}

users = {} # In-memory fallback
mongo_connected = False
users_collection = None
db = None

try:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb+srv://abc:1234@cluster0.jlrvd9l.mongodb.net/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000)
    # Test the connection
    client.admin.command('ping')
    
    db = client['agi-therapist']
    users_collection = db['users']
    mongo_connected = True
    print("[OK] MongoDB connected successfully")
except Exception as e:
    print(f"[WARNING] MongoDB connection failed: {e}")
    print("   -> Switching to LOCAL JSON storage (users.json)")
    mongo_connected = False
    users_collection = None
    users = load_local_users()

# --- GLOBAL MODULES ---
try:
    memory_store = ServerMemoryStore()
    prompt_builder = PromptBuilder(model="Meta-Llama-3.3-70B-Instruct")
    wm = WorkingMemory()
    agent = AGI119Agent()
    safety_engine = EthicalAwarenessEngine()
    style_engine = PersonalizationEngine()
    pers_memory = PersonalizedMemoryModule()
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to initialize global modules: {e}")
    # We might want to exit or set them to None, but app relies on them.
    # For now, let's print and re-raise to see the error in logs
    raise e

# Initialize Clinical Knowledge
try:
    from clinical_resources import get_all_resources
    clinical_data = get_all_resources()
    # memory_store.init_clinical_knowledge(clinical_data)
    print("Skipping Clinical Knowledge Init for debugging")
except Exception as e:
    print(f"[WARNING] Could not initialize clinical knowledge: {e}")




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
        
        # Detect language - check for Hindi (Devanagari) characters
        has_hindi = any('\u0900' <= char <= '\u097f' for char in text)
        tts_lang = 'hi' if has_hindi else 'en'
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

@app.route('/api/dashboard/timeline', methods=['GET'])
@login_required
def dashboard_timeline():
    """Get timeline data for dashboard including session summaries"""
    try:
        user_id = current_user.id
        # Reuse get_conversation_threads logic but format for dashboard
        threads = memory_store.get_conversation_threads(user_id)
        
        timeline_data = []
        for thread in threads:
            # Mock mood for now as it's not strictly stored in metadata yet
            # In a real app, we'd store 'mood' in conversation metadata
            timeline_data.append({
                "id": thread['id'],
                "date": thread['last_message'],
                "title": thread['title'],
                "summary": thread['preview'],
                "message_count": thread['message_count'],
                "mood": "Neutral" # Placeholder, would be fetched from analytics DB
            })
            
        return jsonify({"success": True, "timeline": timeline_data}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/dashboard/report', methods=['GET'])
@login_required
def dashboard_report():
    """Get long-term memory medical/profile report"""
    try:
        user_id = current_user.id
        # Fetch profile memories
        profile_mems = memory_store.retrieve_memories(
            user_id=user_id,
            query="medical history diagnosis medication user profile facts",
            memory_type="profile",
            top_k=20
        )
        
        # Determine "Medical/Clinical" vs "Personal" facts
        medical_facts = []
        personal_facts = []
        
        for mem in profile_mems:
            text = mem['text']
            lower_text = text.lower()
            if any(x in lower_text for x in ['diagnos', 'medic', 'symptom', 'doctor', 'pain', 'disorder']):
                medical_facts.append(text)
            else:
                personal_facts.append(text)
                
        return jsonify({
            "success": True, 
            "report": {
                "medical_history": medical_facts,
                "personal_profile": personal_facts,
                "generated_at": datetime.now().isoformat()
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- GENERATION LOGIC ---
def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript="", conversation_id=None):
    try:
        # Safety check
        if safety_engine.detect_high_risk(transcript):
            print("[WARNING] HIGH RISK DETECTED - Triggering Safety Protocol")
            response_text = safety_engine.ethical_response()
        else:
            # Store user message in working memory (short-term context)
            store_working_memory(user_id, transcript, conversation_id)
            
            # Retrieve memories from both long-term (historical) and working (current session)
            retrieved_bundle = retrieve_memories(user_id, transcript)
            working_context = retrieve_working_memory(user_id, conversation_id)
            
            # Gather reasoning with full context
            reasoning_data = gather_reasoning(user_id, tone, retrieved_bundle, working_context)
            # Build prompt with conversation history
            prompt_data = build_prompt(user_id, transcript, retrieved_bundle, reasoning_data, working_context)
            # Call LLM (Gemini)
            response_text = generate_response_data(perception_result, user_id, transcript, conversation_id)

        audio_path = generate_audio(response_text)
        stored_conversation_id = store_conversation(user_id, transcript, response_text, conversation_id)
        # Store response in working memory
        store_working_memory(user_id, f"Assistant: {response_text}", stored_conversation_id)

        return {"text": response_text, "audio": audio_path, "conversation_id": stored_conversation_id}

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return {"text": "I'm listening. Please go on.", "audio": None, "conversation_id": conversation_id}

def store_working_memory(user_id, message, conversation_id):
    """Store current conversation turn in working memory (short-term)"""
    try:
        working_mem = WorkingMemory(f"working_memory_{conversation_id}")
        # Store just the message string to ensure compatibility with retrieval logic
        working_mem.store(message)
    except Exception as e:
        print(f"[WARNING] Error storing to working memory: {e}")

def retrieve_working_memory(user_id, conversation_id):
    """Retrieve current conversation context from working memory"""
    try:
        working_mem = WorkingMemory(f"working_memory_{conversation_id}")
        all_messages = working_mem.collection.get()
        
        # This guarantees it will be a list, even if the DB explicitly returns None
        documents = all_messages.get("documents") or []
        
        return {"messages": documents, "count": len(documents)}
        
    except Exception as e:
        print(f"[WARNING] Error retrieving working memory: {e}")
        return {"messages": [], "count": 0}

def retrieve_memories(user_id, transcript):
    try:
        # Retrieve profile and episodic memories
        profile_mems = memory_store.retrieve_memories(user_id, transcript, memory_type="profile", top_k=1)
        episodic_mems = memory_store.retrieve_memories(user_id, transcript, memory_type="episodic", top_k=5, recency_days=60)
        
        # Format for display/prompt
        profile_text = "\n".join([m['text'] for m in profile_mems])
        episodic_text = [m['text'] for m in episodic_mems]
        
        return {
            "profile_summary": profile_text,
            "top_memories": episodic_mems, # Keeping full dict for reasoning
            "recency_window": [],
            "risk_flags": []
        }
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        return {"profile_summary": "", "top_memories": [], "recency_window": [], "risk_flags": []}

def gather_reasoning(user_id, tone, retrieved_bundle, working_context=None):
    try:
        user_life = UserLifeUnderstanding(user_id, memory_store=memory_store)
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
            'therapeutic_insight': therapeutic_insight,
            'conversation_history': working_context.get('messages', []) if working_context else []
        }
    except Exception as e:
        print(f"Error gathering reasoning: {str(e)}")
        return {}

def build_prompt(user_id, transcript, retrieved_bundle, reasoning_data, working_context=None):
    try:
        style_config = {"style": "medium", "therapeutic": True}
        prompt_data = prompt_builder.build_prompt(user_id, transcript, retrieved_bundle, style_config, reasoning_data)
        
        # Add conversation history from working memory for context
        if working_context and working_context.get('messages'):
            context_messages = "\nRecent conversation context:\n"
            for msg in working_context['messages'][-5:]:  # Last 5 messages
                context_messages += f"  {msg}\n"
            if isinstance(prompt_data, dict) and 'messages' in prompt_data:
                prompt_data['messages'] = [{
                    "role": "user",
                    "content": context_messages + "\nCurrent message: " + transcript
                }]
        
        return prompt_data
    except Exception as e:
        print(f"Error building prompt: {str(e)}")
        # Fallback with just the current message
        return {
            "messages": [{"role": "user", "content": transcript}], 
            "model": "mixtral-8x7b-32768"
        }

def store_conversation(user_id, transcript, response_text, conversation_id=None):
    try:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # 1. Store in LOGS (Conversation type) for history
        memory_store.store_memory(user_id, "conversation", f"User: {transcript}", 
                                tags=["conversation", "user_message", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id,
                                importance=1.0)
        
        memory_store.store_memory(user_id, "conversation", f"AI: {response_text}", 
                                tags=["conversation", "ai_message", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id,
                                importance=1.0)
        
        # 2. Store in EPISODIC for long-term RAG/Insights
        # [FIX] This ensures past conversations are "remembered" in future sessions
        memory_store.store_memory(user_id, "episodic", f"On {datetime.now().strftime('%Y-%m-%d %H:%M')}, user said: {transcript}", 
                                tags=["episodic", "past_convo", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id,
                                importance=0.8)
        
        memory_store.store_memory(user_id, "episodic", f"Therapist responded: {response_text}", 
                                tags=["episodic", "past_convo", f"conv_{conversation_id}"], 
                                conversation_id=conversation_id,
                                importance=0.5)
        
        return conversation_id
    except Exception as e:
        print(f"Error storing conversation: {str(e)}")
        return None

def _update_env_variable(key: str, value: str, env_path='.env'):
    try:
        # Read existing .env
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write(f"{key}={value}\n")
            return True

        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break

        if not found:
            lines.append(f"{key}={value}\n")

        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"Error updating .env: {e}")
        return False



def update_user_settings(user_id: str, settings_update: dict):
    """Update settings for a user in MongoDB or local storage"""
    try:
        if mongo_connected and users_collection is not None:
            # MongoDB uses dot notation for nested updates
            mongo_update = {f"settings.{k}": v for k, v in settings_update.items()}
            users_collection.update_one({"email": user_id}, {"$set": mongo_update}, upsert=True)
            return True
        else:
            # Update local memory and save to users.json
            u = users.get(user_id)
            if u:
                if not hasattr(u, 'settings'):
                    u.settings = {}
                u.settings.update(settings_update)
                save_local_users(users)
                return True
        return False
    except Exception as e:
        print(f"Error updating user settings: {e}")
        return False

def save_conversation_with_name(user_id, conversation_id, conversation_name):
    """
    Save conversation name/title to long-term memory for historic reference
    """
    try:
        # Store the conversation metadata with name
        conv_metadata = {
            "conversation_id": conversation_id,
            "conversation_name": conversation_name,
            "created_at": datetime.now().isoformat(),
            "user_id": user_id
        }
        
        # Store in MongoDB for quick retrieval
        if mongo_connected and db is not None:
            try:
                conversations_collection = db['conversation_metadata']
                conversations_collection.update_one(
                    {"conversation_id": conversation_id, "user_id": user_id},
                    {"$set": conv_metadata},
                    upsert=True
                )
                print(f"[OK] Conversation saved with name: {conversation_name}")
            except Exception as e:
                print(f"[WARNING] Error saving to MongoDB: {e}")
        
        # Also store a reference in long-term memory for model context
        memory_store.store_memory(
            user_id, 
            "episodic",
            f"Conversation '{conversation_name}' (ID: {conversation_id})",
            tags=["conversation_metadata", f"conv_{conversation_id}"],
            importance=0.8
        )
        
        return {"success": True, "message": f"Conversation saved as '{conversation_name}'"}
    except Exception as e:
        print(f"Error saving conversation with name: {str(e)}")
        return {"success": False, "message": str(e)}


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


@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    try:
        settings = get_user_settings(current_user.id)
        return jsonify({"settings": settings})
    except Exception as e:
        print(f"Error /api/settings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings/gemini-key', methods=['PUT'])
def api_update_gemini_key():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        data = request.get_json() or {}
        api_key = data.get('api_key') or data.get('gemini_api_key')
        if not api_key:
            return jsonify({"success": False, "message": "No API key provided"}), 400

        settings_update = {"gemini_api_key": api_key}
        update_user_settings(current_user.id, settings_update)

        # Update session for immediate use
        session['gemini_api_key'] = api_key
        
        # Also update .env for fallback
        _update_env_variable('GEMINI_API_KEY', api_key)
        os.environ['GEMINI_API_KEY'] = api_key # Synchronize environment
        
        # Re-initialize AAI if possible
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key
        except: pass

        return jsonify({"success": True, "message": "Gemini API key updated"})
    except Exception as e:
        print(f"Error updating gemini key: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/settings/theme', methods=['POST'])
def api_save_theme():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        data = request.get_json() or {}
        # Only persist theme related settings
        dark_mode = bool(data.get('dark_mode', False))
        theme_color = data.get('theme_color', '#667eea')

        settings_update = {
            'theme_color': theme_color,
            'dark_mode': dark_mode
        }

        update_user_settings(current_user.id, settings_update)

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving theme: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/user-history', methods=['GET'])
def api_user_history():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        limit = int(request.args.get('limit', 20))
        history = memory_store.get_conversation_history(current_user.id, limit=limit)
        return jsonify({"success": True, "history": history})
    except Exception as e:
        print(f"Error /api/user-history: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/conversation-threads', methods=['GET'])
def api_conversation_threads():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        threads = memory_store.get_conversation_threads(current_user.id)
        return jsonify({"success": True, "threads": threads})
    except Exception as e:
        print(f"Error /api/conversation-threads: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/conversation/<conversation_id>/messages', methods=['GET'])
def api_conversation_messages(conversation_id):
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        messages = memory_store.get_conversation_messages(current_user.id, conversation_id)
        # Normalize role based on text prefix
        normalized = []
        for m in messages:
            text = m.get('text', '')
            role = 'user' if text.startswith('User:') else ('bot' if text.startswith('AI:') else 'user')
            # Strip prefix
            if text.startswith('User:') or text.startswith('AI:'):
                text = text.split(':', 1)[1].strip()
            normalized.append({'text': text, 'timestamp': m.get('timestamp'), 'role': role})
        return jsonify({"success": True, "messages": normalized})
    except Exception as e:
        print(f"Error /api/conversation/<id>/messages: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/analytics/health', methods=['GET'])
def api_analytics_health():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        # Count condition keywords across conversation history
        conversations = memory_store.get_conversation_history(current_user.id, limit=500)
        
        condition_keywords = {
            "Anxiety": ["anxious", "worried", "nervous", "panic", "fear"],
            "Depression": ["depressed", "sad", "hopeless", "empty", "worthless"],
            "Stress": ["stressed", "overwhelmed", "pressure", "tension"],
            "Insomnia": ["sleep", "insomnia", "can't sleep", "awake"],
            "Trauma": ["trauma", "nightmare", "flashback", "scared"]
        }
        
        counts = {k: 0 for k in condition_keywords.keys()}
        
        for conv in conversations:
            text = conv.get('text', '').lower()
            if text.startswith('user:'): # Only analyze user messages
                for cond, kws in condition_keywords.items():
                    if any(kw in text for kw in kws):
                        counts[cond] += 1
                        
        # Format for Chart.js
        conditions = list(counts.keys())
        data = list(counts.values())
        
        return jsonify({
            "success": True, 
            "labels": conditions,
            "data": data,
            "total_analyzed": len(conversations)
        })
    except Exception as e:
        print(f"Error /api/analytics/health: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/analytics/chat-stats', methods=['GET'])
def api_chat_stats():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    try:
        # Guarantee we get a list back, never None
        conversations = memory_store.get_conversation_history(current_user.id, limit=1000) or []
        total = len(conversations)
        
        user_msgs = sum(1 for c in conversations if c.get('text', '').startswith('User:'))
        ai_msgs = sum(1 for c in conversations if c.get('text', '').startswith('AI:'))
        
        # Rough avg per day
        avg_daily = 0
        if conversations:
            # [FIX] Explicitly cast to string so fromisoformat is 100% safe
            dates = [datetime.fromisoformat(str(c.get('timestamp'))) for c in conversations if c.get('timestamp')]
            
            if dates:
                span_days = max(1, (datetime.now() - min(dates)).days)
                avg_daily = total / span_days

        return jsonify({
            "success": True, 
            "total_messages": total, 
            "user_messages": user_msgs, 
            "ai_messages": ai_msgs, 
            "avg_daily": avg_daily
        })
    except Exception as e:
        print(f"Error /api/analytics/chat-stats: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Safely extract form data (guaranteeing they are strings)
        raw_email = request.form.get('email')
        raw_password = request.form.get('password')
        
        email = str(raw_email).strip().lower() if raw_email else ''
        password = str(raw_password) if raw_password else ''
        
        try:
            user_dict = None
            
            # 1. Try connecting to MongoDB first
            if mongo_connected and users_collection is not None:
                doc = users_collection.find_one({"email": email})
                if doc and isinstance(doc, dict):
                    user_dict = doc
                    
            # 2. Fallback to in-memory dictionary if Mongo fails/is empty
            if not user_dict:
                local_user = users.get(email)
                if local_user:
                    user_dict = {
                        'email': getattr(local_user, 'email', email),
                        'name': getattr(local_user, 'name', email),
                        'password': getattr(local_user, 'password', '')
                    }
            
            # 3. If still nothing, the account doesn't exist
            if not user_dict:
                flash('Account not found')
                return render_template('login.html')
                
            # 4. Safely extract the hashed password
            stored_password = user_dict.get('password') or ""
            
            # 5. Verify password and log the user in
            if check_password_hash(stored_password, password):
                user = User(
                    user_id=user_dict.get('email', email),
                    name=user_dict.get('name', email),
                    email=user_dict.get('email', email),
                    password_hash=stored_password
                )
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Invalid password')
                
        except Exception as e:
            print(f"[ERROR] Login error: {e}")
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
                    "password": password_hash,
                    "settings": get_default_settings()
                })
            else:
                # Store in memory + Disk fallback
                new_user_obj = User(email, name, email, password_hash, settings=get_default_settings())
                users[email] = new_user_obj
                save_local_users(users)
                print(f"📝 User {name} registered in LOCAL STORAGE (users.json)")
                
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
        'max_tokens': 1000,
        'therapeutic_style': 'empathetic',
        'enable_audio': True,
        'theme_color': '#667eea',
        'gemini_api_key': None,
        'usage_count': 0,
        'quota_limit': 100  # Default daily quota
    }

# --- API ROUTES ---
@app.route('/start_conversation', methods=['POST'])
@login_required
def start_conversation():
    try:
        conversation_id = request.form.get('conversation_id') or str(uuid.uuid4())
        greeting = "Hello! I'm your AI therapist. How are you feeling today?"
        
        user_id = current_user.id
        # [NEW] Store the start of the session in long-term memory
        memory_store.store_memory(
            user_id, 
            "conversation", 
            f"AI: {greeting}", 
            tags=["conversation", "ai_message", f"conv_{conversation_id}"], 
            conversation_id=conversation_id
        )
        
        return jsonify({"message": greeting, "type": "bot", "conversation_id": conversation_id})
    except Exception as e:
        print(f"Error in start_conversation: {str(e)}")
        return jsonify({"error": "Error starting conversation"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required. Please log in."}), 401
            
        user_id = current_user.id
        conversation_id = request.form.get('conversation_id')
        
        transcript = get_transcript_from_request()
        if not transcript:
            return jsonify({"error": "No text or audio provided"}), 400
        
        perception_data = analyze_perception(transcript)
        response_data = generate_response_data(perception_data, user_id, transcript, conversation_id)
        
        # Track Usage
        try:
            if mongo_connected and users_collection is not None:
                users_collection.update_one(
                    {"email": user_id},
                    {"$inc": {"settings.usage_count": 1}}
                )
        except Exception as e:
            print(f"Error updating usage stats: {e}")

        # If client is editing an existing message, persist an edit record
        edit_id = request.form.get('edit_id')
        if edit_id:
            try:
                memory_store.store_memory(
                    user_id,
                    "conversation_edit",
                    f"Edit (ref_id={edit_id}): {transcript}",
                    tags=["conversation", "edit", f"conv_{conversation_id}"],
                    conversation_id=str(conversation_id or "default_session"),
                    importance=0.5
                )
            except Exception as e:
                print(f"Warning: could not store edit record: {e}")
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

def get_transcript_from_request():
    """Extract transcript from either text or audio input"""
    # Check for text input first
    if 'text' in request.form and request.form['text'].strip():
        return request.form['text'].strip()
    
    # Check for audio input
    if 'audio' in request.files:
        audio_file = request.files['audio']
        if audio_file and audio_file.filename:
            try:
                # Save audio temporarily with unique name
                ext = os.path.splitext(audio_file.filename)[1] or '.wav'
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4()}{ext}")
                audio_file.stream.seek(0)
                audio_file.save(temp_path)

                # Transcribe audio (the transcribe_audio helper should handle common audio formats)
                transcript = transcribe_audio(temp_path)

                # Clean up
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass

                return transcript
            except Exception as e:
                print(f"[ERROR] Audio transcription failed: {str(e)}")
                return None
    
    return None

def analyze_perception(transcript):
    """Analyze perception from transcript using tone and NLU modules"""
    try:
        # Analyze tone and sentiment
        tone = analyze_tone(transcript)
        
        # Natural Language Understanding
        nlu_result = nlu_process(transcript, tone)
        
        return {
            "transcript": transcript,
            "tone": tone,
            "nlu": nlu_result
        }
    except Exception as e:
        print(f"[WARNING] Perception analysis error: {str(e)}")
        return {
            "transcript": transcript,
            "tone": {"overall_mood": "neutral", "sentiment": "neutral"},
            "nlu": {}
        }

def build_enhanced_prompt_with_perception(transcript, perception_data, reasoning_output):
    """Build ultra-concise prompt combining perception and clinical context."""
    try:
        tone = perception_data.get('tone', {}) if isinstance(perception_data.get('tone'), dict) else {}
        nlu = perception_data.get('nlu', {}) if isinstance(perception_data.get('nlu'), dict) else {}
        
        meta = []
        if tone: meta.append(f"Mood:{tone.get('overall_mood','-')}")
        if nlu and nlu.get('intent'): meta.append(f"Intent:{nlu.get('intent')}")
        
        meta_str = f"[{' | '.join(meta)}]" if meta else ""
        
        # Concise reasoning capped at 250 chars
        reasoning = f"\n[INSIGHT]: {reasoning_output[:250]}" if reasoning_output else ""
            
        return f"{meta_str}{reasoning}"
    except Exception as e:
        return ""
        
        # Check for crisis indicators in transcript
        crisis_keywords = ["suicide", "kill myself", "die", "harm", "hurt myself", "kill myself"]
        is_crisis = any(keyword in transcript.lower() for keyword in crisis_keywords)
        
        # Detect conditions for clinical resources
        condition_keywords = {
            "anxiety": ["anxious", "worried", "nervous", "panic", "fear"],
            "depression": ["depressed", "sad", "hopeless", "empty", "worthless"],
            "stress": ["stressed", "overwhelmed", "pressure"],
            "insomnia": ["sleep", "insomnia", "can't sleep", "lying awake"],
        }
        
        detected_condition = None
        for condition, keywords in condition_keywords.items():
            if any(kw in transcript.lower() for kw in keywords):
                detected_condition = condition
                break
        
        # Get clinical resources
        clinical_context = ""
        if detected_condition:
            strategies = get_coping_strategies(detected_condition)
            if strategies:
                clinical_context = f"\n[CLINICAL RESOURCES] Evidence-based strategies for {detected_condition}:\n"
                for i, strategy in enumerate(strategies[:2], 1):
                    clinical_context += f"  {i}. {strategy}\n"
        
        if is_crisis:
            crisis_res = get_crisis_resources()
            clinical_context += f"\n[CRISIS ALERT] User may be in crisis. Resources: {crisis_res}"
            
        # [NEW] Semantic Clinical Retrieval
        try:
            semantic_clinical = memory_store.retrieve_clinical_knowledge(transcript, top_k=3)
            if semantic_clinical:
                clinical_context += "\n[CLINICAL GUIDELINES & PROTOCOLS]:\n"
                for item in semantic_clinical:
                    clinical_context += f"- {item}\n"
        except Exception as e:
            print(f"Error retrieving semantic clinical info: {e}")
        
        return combined_context + clinical_context
    
    except Exception as e: # type: ignore
            print(f"[WARNING] Error building enhanced prompt: {str(e)}")
            return ""

def generate_response_data(perception_data, user_id, transcript, conversation_id=None):
    """Generate response using Gemini API with clinical integration"""
    
    # [FIX] Initialize api_key safely at start
    api_key = None
    
    try:
        from utils.llm_client import generate_chat_response
        
        # [MEMORY FIX] Store User message in working memory immediately
        if conversation_id and transcript:
            store_working_memory(user_id, f"User: {transcript}", str(conversation_id))

        # Get message history for context
        message_history = []
        smart_context = []
        
        if conversation_id:
            try:
                working_mem = WorkingMemory(f"working_memory_{conversation_id}")
                
                # 1. Sequential History (Pylance safe: guarantees a list, avoiding NoneType slice error)
                sorted_messages = working_mem.get_all_sorted() or []

                # Build sequential history — last 8 messages only to limit tokens
                for msg in sorted_messages[-8:]:  
                    if isinstance(msg, str):
                        if "{'user_id':" in msg: 
                            try:
                                import ast
                                dict_msg = ast.literal_eval(msg)
                                msg = str(dict_msg.get('message', ''))
                            except Exception: 
                                pass
                                
                        if msg.startswith("User:"):
                            message_history.append({"role": "user", "content": msg[5:].strip()})
                        elif msg.startswith("Assistant:"):
                            message_history.append({"role": "assistant", "content": msg[10:].strip()})
                            
                # 2. Semantic retrieval — 2 results, capped at 80 chars each
                # Pylance safe: default to empty dict if retrieve fails
                relevant_docs = working_mem.retrieve(query=str(transcript or ""), n_results=2) or {}
                
                # Pylance safe: extract documents safely, default to empty list if None
                doc_lists = relevant_docs.get('documents') or []
                
                for doc_list in doc_lists:
                    # Pylance safe: ensure the inner list is also iterable
                    for doc in (doc_list or []):
                        doc_str = str(doc)
                        is_in_history = any(doc_str in m.get('content', '') for m in message_history)
                        if not is_in_history and len(doc_str) > 10:
                            smart_context.append(doc_str[:80])  # cap each snippet

            except Exception as e:
                print(f"[WARNING] Could not load message history: {e}")
        
        # [MEMORY INTEGRATION] Retrieve Core Life Insight (max 20 tokens)
        life_facts = ""
        try:
            # Try to get the ultra-concise core insight
            life_facts = memory_store.get_core_insight(user_id)
        except Exception as e:
            print(f"[WARNING] Could not retrieve existing life facts: {e}")
        
        # [FIX] Move API key retrieval UP before it is used for core insight generation
        # Get API key if user has one
        api_key = None
        if mongo_connected and users_collection is not None:
            try:
                user_doc = users_collection.find_one({"email": user_id})
                if user_doc and 'settings' in user_doc:
                    api_key = user_doc['settings'].get('gemini_api_key')
            except Exception as e:
                print(f"[WARNING] Could not get user API key: {e}")
        
        # Use environment API key as fallback
        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # [QUOTA SAVER] Do NOT call generate_core_insight (extra API call).
        # Only use the already-stored core insight, or build from local episodic memories.
        try:
            if not life_facts:
                episodic_mems = memory_store.retrieve_memories(user_id, transcript, memory_type="episodic", top_k=2)
                life_facts = " | ".join([m['text'][:80] for m in episodic_mems])
        except Exception as e:
            print(f"[WARNING] Error in life facts processing: {e}")
        
        # Hard cap on life_facts to prevent system prompt inflation
        if life_facts and len(life_facts) > 200:
            life_facts = life_facts[:200]
        
        # Get reasoning output
        reasoning_output = ""
        try:
            # Run through reasoning modules
            insight_gen = InsightGenerator()
            insights_data = insight_gen.analyze_situation(
                transcript, 
                message_history[-5:] if message_history else [],
                perception_data.get('tone', {}).get('overall_mood', 'neutral'),
                perception_data.get('tone', {}).get('sentiment_score', 0.0)
            )
            
            insights = []
            if isinstance(insights_data, dict):
                 if insights_data.get('recommendation'):
                     insights.append(insights_data.get('recommendation'))
            
            # [LONG-TERM MEMORY INTEGRATION] Retrieve personalized context — capped at 150 chars
            pers_context = pers_memory.get_user_memory_context_formatted(user_id, transcript)
            if pers_context:
                insights.append(pers_context[:150])
            
            reasoning_output = " | ".join(insights) if insights else ""
            # Cap total reasoning output
            if len(reasoning_output) > 250:
                reasoning_output = reasoning_output[:250]
        except Exception as e:
            print(f"[WARNING] Reasoning analysis error: {e}")
        
        # Build enhanced prompt with perception and reasoning
        enhanced_context = build_enhanced_prompt_with_perception(
            transcript, 
            perception_data,
            reasoning_output
        )

        # [SAFETY CHECK] Crisis Intervention Logic
        sentiment_score = perception_data.get('tone', {}).get('sentiment_score', 0.0)
        overall_mood = perception_data.get('tone', {}).get('overall_mood', 'neutral')
        
        # If sentiment is severely negative or specific triggers found
        is_crisis = sentiment_score < -0.7 or "crisis" in reasoning_output.lower() or "suicide" in transcript.lower()
        
        if is_crisis:
            crisis_message = "\n\n[CRISIS PROTOCOL ACTIVATED]"
            crisis_message += "\nIt sounds like you're going through a very difficult time. Please remember I am an AI and cannot provide emergency help."
            crisis_message += "\nIf you are in danger, please contact local emergency services immediately or call/text 988 (Suicide & Crisis Lifeline)."
            
            # Append to context so model knows to handle with care
            enhanced_context += f"\n[SYSTEM ALERT]: User sentiment is critically low ({sentiment_score}). Potential crisis detected. Respond with extreme empathy and resources."
            
            # We will also append resources to the final response text later if needed

        
        # Add Smart Memory Context
        if smart_context:
            enhanced_context += "\n\n[RECALLED CONTEXT from this session]:\n"
            for ctx in smart_context:
                enhanced_context += f"- {ctx}\n"
        
        # Create final message with context
        # Convert message history to format expected by Gemini
        # We need to be careful not to duplicate the LAST user message if we just added it to working memory
        # But for the API call, we construct the prompt normally.
        
        # The user message is the LAST message in the prompt.
        final_user_message = f"{enhanced_context}\n\nUser message: {transcript}"
        
        # Ensure we don't double-add the current user message if it was picked up by WorkingMemory retrieval
        # (Though we added it at the start, so it SHOULD be in message_history[-1] if retrieval worked perfectly)
        # To be safe, we reconstruct the history for the LLM call:
        
        # 1. Take history excluding the very last item if it matches current input (dedup)
        # Cap to last 6 history pairs to keep total payload small
        clean_history = []
        for msg in message_history:
            if msg['content'] != transcript:
                clean_history.append(msg)
        clean_history = clean_history[-6:]  # hard cap: 6 history messages max
                
        # 2. Append the Enhanced User Prompt
        clean_history.append({
            "role": "user",
            "content": final_user_message
        })
        
        # Get LLM response from Gemini
        llm_result = generate_chat_response(
            messages=clean_history,
            api_key=api_key,
            life_facts=life_facts
        )
        
        if llm_result.get("status") == "error":
            response_text = f"I encountered an issue: {llm_result.get('response', 'Unable to generate response')}"
            print(f"[ERROR] LLM Error: {llm_result.get('error')}")
        else:
            response_text = llm_result.get('response', 'I understand.')
        
        # Generate audio response
        audio_path = generate_audio(response_text)
        
        # Store conversation
        stored_conversation_id = store_conversation(user_id, transcript, response_text, conversation_id)
        
        # Store response in working memory
        store_working_memory(user_id, f"Assistant: {response_text}", stored_conversation_id)
        
        response_data = {
            "message": response_text,
            "type": "bot",
            "conversation_id": stored_conversation_id,
            "analysis": perception_data,
            "transcript": transcript if transcript else None
        }
        
        if audio_path:
            filename = os.path.basename(audio_path)
            response_data["audio_url"] = f"/static/audio/{filename}"
        
        # [QUOTA SAVER] Pass None as llm_client so only LOCAL extraction runs (no extra API call)
        current_exchange = f"User: {transcript}\nAI: {response_text}"
        pers_memory.extract_and_save_async(user_id, current_exchange, None, api_key=str(os.environ.get("GEMINI_API_KEY") or ""))

        # [CLINICAL INTELLIGENCE] Async session analytics (emotion + themes + safety)
        try:
            msg_count = len(clean_history)
            clinical_engine.process_session_async(
                user_id=user_id,
                session_id=stored_conversation_id or str(uuid.uuid4()),
                transcript=current_exchange,
                message_count=msg_count
            )
        except Exception as ce:
            print(f"[CLINICAL WARNING] Could not queue session analysis: {ce}")

        return response_data
        
    except Exception as e:
        print(f"[ERROR] Error generating response: {str(e)}")
        return {
            "message": "I'm here to listen. Please go on.",
            "type": "bot",
            "conversation_id": conversation_id,
            "analysis": perception_data,
            "error": str(e)
        }

# --- API ENDPOINTS FOR GEMINI API & CONVERSATIONS ---

@app.route('/api/save-gemini-token', methods=['POST'])
@login_required
def save_gemini_token():
    """Save user's Gemini API key to MongoDB"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        user_id = current_user.id
        
        if not api_key:
            return jsonify({"success": False, "message": "API key cannot be empty"}), 400
        
        if mongo_connected and users_collection is not None:
            try:
                # Update user document with gemini_api_key in settings
                users_collection.update_one(
                    {"email": user_id},
                    {
                        "$set": {
                            "settings.gemini_api_key": api_key,
                            "updated_at": datetime.now().isoformat()
                        }
                    },
                    upsert=True
                )
                session['gemini_api_key'] = api_key  # Also store in session for quick access
                return jsonify({"success": True, "message": "Gemini API key saved successfully"}), 200
            except Exception as e:
                print(f"Error saving Gemini API key: {e}")
                return jsonify({"success": False, "message": f"Error saving API key: {str(e)}"}), 500
        else:
            # Fallback: store in session
            session['gemini_api_key'] = api_key
            return jsonify({"success": True, "message": "Gemini API key saved in session (MongoDB not available)"}), 200
    except Exception as e:
        print(f"Error in save_gemini_token: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/validate-gemini-token', methods=['POST'])
def validate_gemini_token():
    """Validate Gemini API key with comprehensive error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"valid": False, "message": "No data provided", "error": "NO_DATA"}), 400
            
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({"valid": False, "message": "API key cannot be empty", "error": "EMPTY_KEY"}), 400
        
        try:
            # Use the validate function from llm_client
            result = validate_gemini_api_key(api_key)
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {"valid": False, "message": "Validation failed", "error": "VALIDATION_ERROR"}
            
            # Add default fields if missing
            if "error" not in result:
                result["error"] = None if result.get("valid") else "UNKNOWN_ERROR"
            
            # Store valid key in session
            if result.get("valid"):
                session['gemini_api_key'] = api_key
            
            return jsonify(result), 200
        except Exception as e:
            print(f"[ERROR] Validation error: {e}")
            return jsonify({"valid": False, "message": f"Validation failed: {str(e)}", "error": "VALIDATION_FAILED"}), 500
            
    except ValueError as e:
        return jsonify({"valid": False, "message": f"Invalid JSON: {str(e)}", "error": "INVALID_JSON"}), 400
    except Exception as e:
        print(f"Error in validate_gemini_token: {e}")
        return jsonify({"valid": False, "message": f"Unexpected error: {str(e)}", "error": "UNEXPECTED_ERROR"}), 500

@app.route('/api/conversations', methods=['GET'])
@login_required
def get_conversations():
    """Get all conversations for the current user using optimized memory retrieval"""
    try:
        user_id = current_user.id
        threads = memory_store.get_conversation_threads(user_id)
        
        # Adapt thread format to existing frontend expectations
        formatted_convos = []
        for thread in threads:
            formatted_convos.append({
                "id": thread['id'],
                "timestamp": thread['last_message'],
                "preview": thread['preview'] + "..." if len(thread['preview']) >= 100 else thread['preview'],
                "title": thread.get('title', f"Session {thread['id'][:8]}")
            })
            
        return jsonify({
            "success": True,
            "conversations": formatted_convos # Already sorted by get_conversation_threads
        }), 200
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        return jsonify({"success": False, "conversations": [], "message": str(e)}), 500
            


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
@login_required
def get_conversation_detail(conversation_id):
    """Get detailed messages from a conversation using optimized retrieval"""
    try:
        user_id = current_user.id
        messages = memory_store.get_conversation_messages(user_id, conversation_id)
        
        formatted_messages = []
        for msg in messages:
            text = msg.get('text', '')
            role = "user" if text.startswith("User:") else "assistant"
            # Clean text
            clean_text = text.replace("User: ", "").replace("AI: ", "").replace("Assistant: ", "")
            
            formatted_messages.append({
                "role": role,
                "text": clean_text,
                "timestamp": msg.get('timestamp')
            })
            
        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "messages": formatted_messages
        }), 200
    except Exception as e:
        print(f"Error fetching conversation detail: {e}")
        return jsonify({"success": False, "messages": [], "message": str(e)}), 500
            

@app.route('/api/save-conversation-with-name', methods=['POST'])
@login_required
def save_conversation_with_name_endpoint():
    """Save conversation with a custom name to long-term memory"""
    try:
        user_id = current_user.id
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        conversation_id = data.get('conversation_id', '').strip()
        conversation_name = data.get('conversation_name', '').strip()
        
        if not conversation_id:
            return jsonify({"success": False, "message": "Conversation ID is required"}), 400
        
        if not conversation_name:
            return jsonify({"success": False, "message": "Conversation name is required"}), 400
        
        if len(conversation_name) > 255:
            return jsonify({"success": False, "message": "Conversation name is too long (max 255 characters)"}), 400
        
        # Save conversation with name
        result = save_conversation_with_name(user_id, conversation_id, conversation_name)
        
        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        print(f"Error in save_conversation_with_name_endpoint: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/get-named-conversations', methods=['GET'])
@login_required
def get_named_conversations():
    """Get all saved conversations with names from long-term memory"""
    try:
        user_id = current_user.id
        
        # Retrieve from MongoDB conversation_metadata
        if mongo_connected and db is not None:
            try:
                conversations_collection = db['conversation_metadata']
                conversations = list(conversations_collection.find(
                    {"user_id": user_id},
                    {"_id": 0, "conversation_id": 1, "conversation_name": 1, "created_at": 1}
                ).sort("created_at", -1))
                
                return jsonify({
                    "success": True,
                    "conversations": conversations,
                    "count": len(conversations)
                }), 200
            except Exception as e:
                print(f"[WARNING] Error retrieving from MongoDB: {e}")
                return jsonify({
                    "success": False,
                    "message": f"Could not retrieve conversations: {str(e)}"
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": "MongoDB not available"
            }), 503
            
    except Exception as e:
        print(f"Error in get_named_conversations: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/delete-conversation', methods=['POST'])
@login_required
def api_delete_conversation():
    """Delete a conversation by id from both metadata and long-term memory"""
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id', '').strip()
        if not conversation_id:
            return jsonify({"success": False, "message": "conversation_id is required"}), 400

        user_id = current_user.id

        # Delete metadata from MongoDB if present
        if mongo_connected and db is not None:
            try:
                conversations_collection = db['conversation_metadata']
                conversations_collection.delete_many({"conversation_id": conversation_id, "user_id": user_id})
            except Exception as e:
                print(f"[WARNING] Could not delete conversation metadata: {e}")

        # Delete from Chroma via MemoryStore
        success = memory_store.delete_conversation(user_id, conversation_id)
        
        # [NEW] Also clean up the temporary working memory collection
        try:
            wm = WorkingMemory(f"working_memory_{conversation_id}")
            wm.clear()
        except Exception as e:
            print(f"[WARNING] Could not clear working memory for {conversation_id}: {e}")
        
        if success:
            print(f"[OK] Conversation {conversation_id} deleted successfully for {user_id}")
            return jsonify({"success": True, "message": "Conversation deleted successfully"}), 200
        else:
            print(f"[ERROR] Failed to delete conversation {conversation_id} for {user_id}")
            return jsonify({"success": False, "message": "Failed to delete conversation from memory store"}), 500
            
    except Exception as e:
        print(f"Error in api_delete_conversation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/conversation-context/<conversation_id>', methods=['GET'])
@login_required
def get_conversation_context(conversation_id):
    """Get conversation context from both working and long-term memory for AI context"""
    try:
        user_id = current_user.id
        
        # Get from working memory if conversation is recent
        working_context = retrieve_working_memory(user_id, conversation_id)
        
        # Get from long-term memory
        try:
            ltm_messages = memory_store.retrieve_memories(
                user_id=user_id,
                query="",
                top_k=20,
                filter_tags=[f"conv_{conversation_id}"]
            )
            ltm_text = "\n".join([mem.get('text', '') for mem in ltm_messages])
        except Exception as e:
            print(f"Error retrieving LTM: {e}")
            ltm_text = ""
        
        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "working_memory": working_context,
            "long_term_memory": ltm_text,
            "history_count": len(working_context.get('messages', [])) or len(ltm_text.split('\n'))
        }), 200
    except Exception as e:
        print(f"Error in get_conversation_context: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/memory-context', methods=['GET'])
@login_required
def get_memory_context():
    """Get general memory context about the user (historical insights)"""
    try:
        user_id = current_user.id
        
        # Get user profile summary and top memories
        try:
            user_memories = memory_store.retrieve_memories(
                user_id=user_id,
                query="",
                top_k=20,
                filter_tags=["profile", "episodic"]
            )
            
            memory_summary = {
                "total_memories": len(user_memories),
                "key_memories": [mem.get('text', '') for mem in user_memories[:5]],
                "profile": memory_store.get_profile(user_id) if hasattr(memory_store, 'get_profile') else None
            }
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            memory_summary = {"total_memories": 0, "key_memories": [], "profile": None}
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "memory_summary": memory_summary
        }), 200
    except Exception as e:
        print(f"Error in get_memory_context: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/chat-memory', methods=['POST'])
@login_required
def save_chat_message():
    """Save individual chat message to both working and long-term memory"""
    try:
        user_id = current_user.id
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        conversation_id = data.get('conversation_id', '').strip()
        message_text = data.get('message', '').strip()
        is_user = data.get('is_user', True)
        
        if not conversation_id or not message_text:
            return jsonify({"success": False, "message": "Missing conversation_id or message"}), 400
        
        # Store in working memory
        store_working_memory(user_id, message_text, conversation_id)
        
        # Store in long-term memory
        sender = "User" if is_user else "AI"
        
        # 1. Store as Log (for history display)
        memory_store.store_memory(
            user_id,
            "conversation",
            f"{sender}: {message_text}",
            tags=["conversation", f"conv_{conversation_id}"],
            conversation_id=conversation_id,
            importance=1.0
        )

        # 2. Store as Episodic (for RAG/Intelligence)
        memory_store.store_memory(
            user_id,
            "episodic",
            f"Message in {conversation_id}: {sender}: {message_text}",
            tags=["episodic", "message", f"conv_{conversation_id}"],
            conversation_id=conversation_id,
            importance=0.7
        )
        
        return jsonify({"success": True, "message": "Message saved to memory"}), 200
    except Exception as e:
        print(f"Error in save_chat_message: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/personalized-memory-context/<user_id>', methods=['GET'])
@login_required
def api_get_user_memory_context(user_id):
    """Retrieve structured long-term personalized memory summary for a user (legacy)"""
    if current_user.id != user_id:
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    try:
        report = pers_memory.get_full_memory_report(user_id)
        return jsonify(report), 200
    except Exception as e:
        print(f"Error in api_get_user_memory_context: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/memory/sync-history', methods=['POST'])
@login_required
def api_sync_history():
    """Trigger bulk extraction from historical conversations"""
    user_id = current_user.id
    try:
        # 1. Get all conversation threads
        threads = memory_store.get_conversation_threads(user_id)
        if not threads:
            return jsonify({"success": True, "message": "No historical conversations found to sync."}), 200
        
        # 2. Reconstruct full conversation data
        all_convos = []
        for thread in threads:
            conv_id = thread['id']
            # Get messages for this specific conversation
            messages = memory_store.get_conversation_messages(user_id, conv_id)
            all_convos.append({
                "conversation_id": conv_id,
                "messages": [{"role": "user" if m['text'].startswith("User:") else "assistant", "text": m['text'].replace("User: ", "").replace("AI: ", "").replace("Assistant: ", "")} for m in messages]
            })
        
        # 3. Use thread for bulk analysis to avoid blocking
        def run_sync():
            try:
                from utils.llm_client import generate_chat_response
                api_key = session.get('gemini_api_key') or os.environ.get("GEMINI_API_KEY")
                pers_memory.analyze_historical_data(user_id, all_convos, generate_chat_response, api_key=str(os.environ.get("GEMINI_API_KEY") or ""))
            except Exception as e:
                print(f"[MEMORY SYNC ERROR] {e}")

        import threading
        threading.Thread(target=run_sync).start()
        
        return jsonify({
            "success": True, 
            "message": f"Historical sync started for {len(threads)} sessions in the background. Your long-term memory will be populated shortly."
        }), 200
        
    except Exception as e:
        print(f"Error in api_sync_history: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

        return jsonify({"success": False, "message": str(e)}), 500

# [NEW] Dashboard Routes

@app.route('/api/dashboard/timeline', methods=['GET'])
@login_required
def api_dashboard_timeline():
    """Get enriched conversation timeline for dashboard (Safiya's feature)"""
    user_id = current_user.id
    try:
        # Pull memory store threads
        threads = memory_store.get_conversation_threads(user_id)

        # Pull clinical analytics for emotion enrichment
        clinical_sessions = clinical_engine.store.get_user_sessions(user_id, limit=200)
        emotion_map = {s.get('session_id'): s for s in clinical_sessions}

        timeline = []
        for t in threads:
            sid = t.get('id', '')
            clin = emotion_map.get(sid, {})
            # Parse stored themes JSON
            try:
                themes = json.loads(clin.get('themes', '[]')) if clin else []
            except Exception:
                themes = []
            timeline.append({
                "id":            sid,
                "title":         t.get('title', 'Therapy Session'),
                "date":          t.get('last_message'),
                "summary":       t.get('preview', 'Session recorded and analysed.'),
                "message_count": t.get('message_count', clin.get('message_count', 0)),
                "mood":          clin.get('emotion', t.get('mood', 'neutral')),
                "confidence":    clin.get('confidence', 0.0),
                "themes":        themes,
                "risk_flag":     bool(clin.get('risk_flag', False)),
            })

        # Also include clinical sessions that don't have a thread
        thread_ids = {t['id'] for t in threads}
        for s in clinical_sessions:
            if s.get('session_id') not in thread_ids:
                try:
                    themes = json.loads(s.get('themes','[]'))
                except Exception:
                    themes = []
                timeline.append({
                    "id":            s.get('session_id',''),
                    "title":         f"Session — {s.get('timestamp','')[:10]}",
                    "date":          s.get('timestamp',''),
                    "summary":       "Chat session analysed by clinical intelligence.",
                    "message_count": s.get('message_count', 0),
                    "mood":          s.get('emotion','neutral'),
                    "confidence":    s.get('confidence', 0.0),
                    "themes":        themes,
                    "risk_flag":     bool(s.get('risk_flag', False)),
                })

        # Sort newest first
        timeline.sort(key=lambda x: x.get('date') or '', reverse=True)
        return jsonify({"success": True, "timeline": timeline}), 200
    except Exception as e:
        print(f"[ERROR] /api/dashboard/timeline: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/dashboard/report', methods=['GET'])
@login_required
def api_dashboard_report():
    """Get medical and personal report for dashboard — merges LLM memory + regex extractor"""
    user_id = current_user.id
    try:
        # 1. Try LLM-extracted personalized memory first
        raw_report = pers_memory.get_full_memory_report(user_id)

        medical_profile = []
        for item in raw_report.get('medical_flags', []):
            medical_profile.append(f"🏥 {item.get('key','Medical')}: {item.get('value','')}")
        for item in raw_report.get('risk_indicators', []):
            medical_profile.append(f"⚠️ Risk: {item.get('key','Risk')}: {item.get('value','')}")

        personal_profile = []
        for item in raw_report.get('important_identity', []):
            personal_profile.append(f"👤 {item.get('key','')}: {item.get('value','')}")
        for item in raw_report.get('life_story_events', []):
            personal_profile.append(f"📌 Event: {item.get('value','')}")
        for item in raw_report.get('recurring_themes', []):
            personal_profile.append(f"🔄 Theme: {item.get('value','')}")

        # 2. Augment with regex-extracted clinical data from transcripts
        clinical_report = clinical_engine.get_medical_report(user_id)
        extra_medical   = clinical_report.get("medical_history", [])
        extra_personal  = clinical_report.get("personal_profile", [])
        risk_flags      = clinical_report.get("risk_flags", [])

        if extra_medical and "No medical records detected yet" not in extra_medical[0]:
            for item in extra_medical:
                if item not in medical_profile:
                    medical_profile.append(item)

        if extra_personal and "No personal profile data detected yet" not in extra_personal[0]:
            for item in extra_personal:
                if item not in personal_profile:
                    personal_profile.append(item)

        if not medical_profile:
            medical_profile = ["No medical records found. Keep chatting — they will appear automatically."]
        if not personal_profile:
            personal_profile = ["No personal profile data yet. As you chat, your profile builds automatically."]

        return jsonify({
            "success": True,
            "report": {
                "medical_history":  medical_profile,
                "personal_profile": personal_profile,
                "risk_flags":       risk_flags,
            }
        }), 200
    except Exception as e:
        print(f"[ERROR] /api/dashboard/report: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ─── Clinical Analytics Routes ────────────────────────────────────────────

@app.route('/analytics')
@login_required
def analytics_page():
    """Dedicated full-page analytics dashboard"""
    return render_template('analytics.html')


@app.route('/api/user-therapy-analytics', methods=['GET'])
@login_required
def api_user_therapy_analytics():
    """
    GET /api/user-therapy-analytics
    Returns complete dashboard JSON for the logged-in user.
    Includes: anxiety_trend, mood_stability, TPS, EVI,
              dominant_stressor, topic_frequency, risk_alerts, etc.
    """
    try:
        user_id = current_user.id
        data = clinical_engine.get_dashboard_data(user_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"[ERROR] /api/user-therapy-analytics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/user-memory-context/<user_id>', methods=['GET'])
@login_required
def api_user_memory_context_clinical(user_id):
    """
    GET /api/user-memory-context/<user_id>
    Returns structured long-term memory summary (clinical analytics version).
    Also merges the standard personalized memory report.
    """
    if current_user.id != user_id:
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    try:
        # Clinical analytics summary
        clinical_ctx = clinical_engine.get_user_memory_context(user_id)
        # Personalized memory (identity / medical / themes)
        mem_report = pers_memory.get_full_memory_report(user_id)
        return jsonify({
            "success": True,
            "clinical_summary": clinical_ctx,
            "memory_report": mem_report
        }), 200
    except Exception as e:
        print(f"[ERROR] /api/user-memory-context/<user_id>: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/user-risk-alerts/<user_id>', methods=['GET'])
@login_required
def api_user_risk_alerts(user_id):
    """
    GET /api/user-risk-alerts/<user_id>
    Returns all risk alerts for the user (suicidal ideation, self-harm, etc.).
    """
    if current_user.id != user_id:
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    try:
        data = clinical_engine.get_risk_alerts(user_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"[ERROR] /api/user-risk-alerts/<user_id>: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)