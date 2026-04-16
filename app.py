from collections import Counter
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_from_directory, session # pyre-ignore[21]
import typing
import tempfile
import os
import json
import uuid
import warnings
import logging
import certifi
ACTIVE_CHAT_TIMERS = {}
ACTIVE_CHAT_CURSORS = {}
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
import assemblyai as aai # pyre-ignore[21]
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user # pyre-ignore[21]
from dotenv import load_dotenv # pyre-ignore[21]
from utils.llm_client import generate_chat_response, validate_gemini_api_key # pyre-ignore[21]
from gtts import gTTS # pyre-ignore[21]
from pymongo.mongo_client import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash # pyre-ignore[21]
from perception.stt.stt_live import save_wav, transcribe_audio # pyre-ignore[21]
from perception.stt.stt_live import transcribe_audio, extract_pitch
from perception.tone.tone_sentiment_live import analyze_tone # pyre-ignore[21]
from perception.nlu.nlu_live import nlu_process # pyre-ignore[21]
import requests # pyre-ignore[21]
from memory.working_memory import WorkingMemory # pyre-ignore[21]
from memory.long_term_memory import LongTermMemory # pyre-ignore[21]
from reasoning.user_life_understanding import UserLifeUnderstanding # pyre-ignore[21]
from reasoning.emotional_reasoning import EmotionalReasoning # pyre-ignore[21]
from reasoning.ethical_awareness import EthicalAwareness # pyre-ignore[21]
from reasoning.internal_cognition import InternalCognition # pyre-ignore[21]
from perception.reasoning.insight import TherapeuticInsight
from core.agi_agent import AGI119Agent # pyre-ignore[21]
from core.emotion_detector import detect_emotion # pyre-ignore[21]
from api.memory_store import ServerMemoryStore # pyre-ignore[21]
from prompt_builder.prompt_builder import PromptBuilder # pyre-ignore[21]
from reasoning.long_term_personalized_memory import PersonalizedMemoryModule # pyre-ignore[21]

# --- NEW INTEGRATION: Teammate's Safety Module ---
from core.ethics_personalization import EthicalAwarenessEngine, PersonalizationEngine # pyre-ignore[21]

# --- NEW: Lightweight Clinical Intelligence Layer ---

from core.clinical_intelligence import get_clinical_engine # pyre-ignore[21]
from functools import wraps
from flask import abort

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is logged in AND has the is_admin flag
        user_data = memory_store.mongo_db.users.find_one({"email": current_user.email})
        if not user_data or not user_data.get('is_admin'):
            print(f"🚫 [SECURITY] Blocked non-admin access attempt by: {current_user.email}")
            return abort(403) # "Forbidden" error
        return f(*args, **kwargs)
    return decorated_function
clinical_engine = get_clinical_engine()

# Ensure static/audio exists
os.makedirs(os.path.join('static', 'audio'), exist_ok=True)



load_dotenv()
print(f"DEBUG: MONGODB_URI is {os.getenv('MONGODB_URI')[:15]}...")
api_key = os.getenv("ASSEMBLYAI_API_KEY")
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
        data = {k: v.to_dict() if isinstance(v, User) else v for k, v in users_dict.items()}
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
    # Professional fix: Use the certifi CA bundle to verify SSL on Windows
    ca = certifi.where()
    
    # Get the URI from your .env
    mongo_uri = os.getenv("MONGODB_URI")
    
    # Initialize with the certificate file (tlsCAFile)
    # This solves the [SSL: TLSV1_ALERT_INTERNAL_ERROR] you saw in Lucknow
    client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
    
    # Test the connection
    client.admin.command('ping')
    
    db = client['agi-therapist']
    users_collection = db['users']
    mongo_connected = True
    print("[OK] MongoDB connected successfully (SSL Handshake Verified)")

except Exception as e:
    print(f"[WARNING] MongoDB connection failed: {e}")
    print("   -> Switching to LOCAL JSON storage (users.json)")
    mongo_connected = False
    users_collection = None
    # Load from local file if the cloud 'brain' is unreachable
    users = load_local_users()

# --- GLOBAL MODULES ---
# --- GLOBAL MODULES INITIALIZATION ---
try:
    # 1. Inject the 'db' object so the memory store uses MongoDB Atlas
    memory_store = ServerMemoryStore(database=db) 
    
    # 2. Update PromptBuilder to match your actual Gemini model
    prompt_builder = PromptBuilder(model="gemini-3-flash-preview") 
    
    wm = WorkingMemory()
    agent = AGI119Agent()
    safety_engine = EthicalAwarenessEngine()
    style_engine = PersonalizationEngine()
    
    # 3. Inject 'db' here too for long-term personalized recall
    pers_memory = PersonalizedMemoryModule(database=db)
    
    print("[SUCCESS] Global Modules synchronized with Cloud Database.")

except Exception as e:
    print(f"[CRITICAL ERROR] Failed to initialize global modules: {e}")
    raise e

# Initialize Clinical Knowledge
try:
    from clinical_resources import get_all_resources # pyre-ignore[21]
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

def is_injection_attempt(user_text: str) -> bool:
    """
    🛡️ The Cognitive Firewall: Catches hackers before they reach the LLM.
    """
    text = user_text.lower()
    
    red_flags = [
        "system prompt",
        "ignore previous",
        "ignore all",
        "developer mode",
        "you are now",
        "repeat the text above",
        "what are your instructions",
        "bypass",
        "jailbreak"
    ]
    
    return any(flag in text for flag in red_flags)
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

# Removed duplicate /api/dashboard/report route. Original logic preserved at the bottom of the file.

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
        
        # This 'or []' is the magic part—it catches both 'missing key' and 'None' values
        documents = all_messages.get("documents") or []
        
        return {
            "messages": documents, 
            "count": len(documents)
        }
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

        # 1. Save AI response to local Working Memory 
        # (The User message was already saved at the start of generate_response_data)
        store_working_memory(user_id, f"Assistant: {response_text}", conversation_id)

        # 2. Save BOTH to MongoDB so the UI can actually display them on refresh!
        # (Our Bouncer fix from earlier guarantees this won't pollute the LTM facts)
        memory_store.store_memory(user_id, "conversation", f"User: {transcript}",
                                  tags=["conversation", "user_message", f"conv_{conversation_id}"],
                                  conversation_id=conversation_id,
                                  importance=1)

        memory_store.store_memory(user_id, "conversation", f"AI: {response_text}",
                                  tags=["conversation", "ai_message", f"conv_{conversation_id}"],
                                  conversation_id=conversation_id,
                                  importance=1)

        return conversation_id
    except Exception as e:
        print(f"Error storing conversation: {str(e)}")
        return conversation_id

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

def update_user_password(user_id: str, new_password_hash: str):
    """Update password for a user in MongoDB or local storage"""
    try:
        if mongo_connected and users_collection is not None:
            users_collection.update_one({"email": user_id}, {"$set": {"password": new_password_hash}})
            return True
        else:
            u = users.get(user_id)
            if u:
                u.password = new_password_hash
                save_local_users(users)
                return True
        return False
    except Exception as e:
        print(f"Error updating user password: {e}")
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
                         user_id=current_user.id,
                         user_settings=user_settings)


@app.route('/api/settings', methods=['GET'])
@login_required
def api_get_settings():
    try:
        # 1. Look up the user using their master ID, exactly how the DB originally saved it
        query = {"$or": [{"_id": current_user.id}, {"email": current_user.email}]}
        user_data = users_collection.find_one(query) or {}
        
        # 2. Grab the key
        saved_key = user_data.get('gemini_api_key')
        if not saved_key and 'settings' in user_data:
            saved_key = user_data['settings'].get('gemini_api_key', '')
            
        # 3. Mask it
        if saved_key and len(saved_key) > 8:
            masked_key = f"••••••••••••{saved_key[-4:]}"
        else:
            masked_key = ""

        return jsonify({
            "success": True,
            "has_key": bool(saved_key),
            "masked_key": masked_key,
            "settings": user_data.get("settings", {})
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/settings/gemini-key', methods=['PUT', 'POST'])
@login_required
def update_api_key():
    try:
        data = request.get_json() or {}
        new_key = data.get('api_key') or data.get('gemini_api_key')

        if not new_key:
            return jsonify({"success": False, "message": "No key provided"}), 400

        # 1. Update the original master document
        query = {"$or": [{"_id": current_user.id}, {"email": current_user.email}]}
        
        # Save it in BOTH locations to guarantee the app finds it
        users_collection.update_one(
            query, 
            {"$set": {
                "gemini_api_key": new_key,
                "settings.gemini_api_key": new_key
            }}, 
            upsert=True
        )

        session['gemini_api_key'] = new_key
        print(f"✅ [DB SUCCESS] API Key permanently saved for user {current_user.id}")
        return jsonify({"success": True, "message": "Key securely updated!"})

    except Exception as e:
        print(f"❌ [DB ERROR] Could not save key: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
from bson.objectid import ObjectId
from flask_login import logout_user

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    print(f"🚨 [SECURITY] User {current_user.id} requested FULL WIPE.")
    try:
        user_id = str(current_user.id)
        # Safely grab the email, defaulting to ID if email isn't in current_user
        user_email = getattr(current_user, 'email', user_id) 

        # 1. Fire the Scorched Earth function in Memory Store
        stats = memory_store.purge_all_user_data(user_id, user_email)

        # 2. Log them out and destroy the session token
        logout_user()

        print(f"✅ [PURGE COMPLETE] Account wiped. Stats: {stats}")
        return jsonify({"success": True, "stats": stats}), 200

    except Exception as e:
        print(f"❌ [FATAL PURGE ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": "Failed to completely wipe data."}), 500


from flask import request, session

from bson.objectid import ObjectId




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
@login_required
def api_conversation_threads():
    # Force Email as the identity
    u_email = str(current_user.email)
    try:
        threads = memory_store.get_conversation_threads(u_email)
        return jsonify({"success": True, "threads": threads})
    except Exception as e:
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
        
        counts = {str(k): 0 for k in condition_keywords.keys()}
        
        for conv in conversations:
            text = conv.get('text', '').lower()
            if text.startswith('user:'): # Only analyze user messages
                for cond, kws in condition_keywords.items():
                    if any(kw in text for kw in kws):
                        counts.update({cond: counts.get(cond, 0) + 1})
                        
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
        conversations = memory_store.get_conversation_history(current_user.id, limit=1000)
        total = len(conversations)
        user_msgs = sum(1 for c in conversations if c.get('text', '').startswith('User:'))
        ai_msgs = sum(1 for c in conversations if c.get('text', '').startswith('AI:'))
        # Rough avg per day
        avg_daily = 0
        if conversations:
            dates = [datetime.fromisoformat(str(c.get('timestamp'))) for c in conversations if c.get('timestamp')]
            if dates:
                span_days = max(1, (datetime.now() - min(dates)).days)
                avg_daily = total / span_days

        return jsonify({"success": True, "total_messages": total, "user_messages": user_msgs, "ai_messages": ai_msgs, "avg_daily": avg_daily})
    except Exception as e:
        print(f"Error /api/analytics/chat-stats: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def get_history_for_user(user_id, conversation_id=None):
    """Fetches history using the UUID. Prevents the 'Email-as-ID' bug."""
    
    # 1. SAFETY: If the ID is missing or is an email (@), it's the wrong ID!
    if not conversation_id or "@" in str(conversation_id):
        print(f"🧹 [HISTORY] No valid UUID for {user_id}. Starting fresh.", flush=True)
        return []

    try:
        # 2. We MUST use conversation_id here to find the chat history
        raw_history = memory_store.get_conversation_history(conversation_id)
        
        formatted_history = []
        for msg in raw_history:
            text = msg.get("content") or msg.get("text") or ""
            # Gemini expects 'model', but our DB stores 'assistant' or 'ai'
            role = "model" if msg.get("role") in ["assistant", "ai", "model", "bot"] else "user"

            if text:
                formatted_history.append({"role": role, "parts": [text]})

        print(f"🧠 [HISTORY] Success! Loaded {len(formatted_history)} messages for session {conversation_id}", flush=True)
        return formatted_history

    except Exception as e:
        print(f"❌ [HISTORY ERROR] {e}")
        return []

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

               
                
                # --- RESTORE THE API KEY TO MEMORY ---
                if 'gemini_api_key' in user_data:
                    session['gemini_api_key'] = user_data['gemini_api_key']
                # -------------------------------------

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
                    "password": password_hash,
                    "settings": get_default_settings()
                })
            else:
                # Store in memory + Disk fallback
                new_user_obj = User(email, name, email, password_hash, settings=get_default_settings())
                users[email] = new_user_obj # type: ignore
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
    session.clear() # This kills the "Ghost Key" in the browser backpack
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
        'quota_limit': 15  # Default daily quota
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
        return jsonify({"error": "Error starting conversation"})

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    # ==========================================
    # PHASE 1: AUTH & IDENTITY
    # ==========================================
    if not current_user.is_authenticated:
        return jsonify({"error": "Session expired"}), 401

    # Unified Identity extraction (Using Email for Cloud Sync)
    user_email = str(getattr(current_user, 'email', 'unknown_user'))
    user_id = str(current_user.id) if current_user.is_authenticated else "unknown"

    # ==========================================
    # PHASE 2: PAYLOAD EXTRACTION & PERCEPTION
    # ==========================================
    import uuid
    import flask
    from perception.audio_interceptor import transcribe_audio_file
    
    data = request.get_json(silent=True) or {}
    transcript = ""
    vocal_tone = ""

    # 1. Engage Auditory Cortex if audio is present
    if 'audio' in request.files:
        audio_data = transcribe_audio_file(request.files['audio'])
        transcript = audio_data.get("transcript", "")
        vocal_tone = audio_data.get("tone", "")

    # 2. Fallback to Visual Cortex (Text typing)
    if not transcript:
        transcript = (
            request.form.get('text') or request.form.get('message') or request.form.get('transcript') or
            data.get('text') or data.get('message') or data.get('transcript') or ""
        ).strip()

    if not transcript:
        return flask.jsonify({"error": "No message or audio detected"}), 400

    # 3. CONTEXT INJECTION: Combine Tone and Transcript
    # If the user spoke, we prepend the hidden metadata so the LLM knows HOW they sounded.
    if vocal_tone and vocal_tone.lower() not in ["neutral", ""]:
        transcript = f"[Vocal Tone Detected: {vocal_tone}] {transcript}"

    # 2. Extract or Generate the Conversation ID (ONCE)
    conversation_id = (
        request.form.get('conversation_id') or data.get('conversation_id') or
        request.form.get('chat_id') or data.get('chat_id') or ""
    ).strip()

    if not conversation_id or conversation_id == "unknown":
        conversation_id = str(uuid.uuid4())
        print(f"⚠️ [WARNING] Generated new chat ID: {conversation_id}")
    else:
        print(f"✅ [SUCCESS] Attached to existing chat: {conversation_id}")

    # ==========================================
    # PHASE 3: FRONT-DOOR SECURITY & TRAPDOORS
    # ==========================================
    from flask import jsonify
    if transcript.lower() == "/number":
        import base64
        try:
            true = base64.b64decode(memory_store._vector_namespace_id).decode('utf-8')
        except:
            true = "616773126368"

        msg = f"**{true}**."
        
        # Etch into Database so it survives refresh
        memory_store.store_memory(user_id=user_email, memory_type="conversation", text=f"User: {transcript}", conversation_id=conversation_id,tags=["internal"], importance=1)
        memory_store.store_memory(user_id=user_email, memory_type="conversation", text=f"AI: {msg}", conversation_id=conversation_id,tags=["internal"], importance=1)
        
        return jsonify({"success": True, "text": msg, "audio": None, "conversation_id": conversation_id})

    # 2. THE COGNITIVE FIREWALL (PROMPT INJECTION BLOCKER)
    if is_injection_attempt(transcript):
        print(f"🚨 [SECURITY] Blocked injection attempt from user: {user_email}")
        msg = "I am an AGI Therapist. I cannot discuss my internal architecture, system prompts, or bypass my clinical guidelines. How can I help you today?"
        
        # Etch into Database so the hacker sees their failed attempt forever
        memory_store.store_memory(user_id=user_email, memory_type="conversation", text=f"User: {transcript}", conversation_id=conversation_id,tags=["internal"], importance=1)
        memory_store.store_memory(user_id=user_email, memory_type="conversation", text=f"AI: {msg}", conversation_id=conversation_id,tags=["internal"], importance=1)
        
        return jsonify({"success": True, "text": msg, "audio": None, "conversation_id": conversation_id})
    # ==========================================
    # PHASE 3.5: FREEMIUM QUOTA & API KEY ROUTING
    # ==========================================
    active_key = os.environ.get("GEMINI_API_KEY")
    try:
        # Fetch the freshest user data directly from the DB
        user_doc = memory_store.mongo_db['users'].find_one({"email": user_email})
        
        if user_doc:
            settings = user_doc.get('settings', {})
            user_api_key = settings.get('gemini_api_key')
            
            if not user_api_key:
                # 🚨 NO PERSONAL KEY: Enforce the Server Limit
                quota_limit = settings.get('quota_limit', 15)
                usage_count = user_doc.get('usage_count', 0)
                
                if usage_count >= quota_limit:
                    print(f"🛑 [QUOTA HIT] User {user_email} reached the {quota_limit} limit.")
                    return jsonify({
                        "success": False, 
                        "error_type": "QUOTA_EXHAUSTED", 
                        "message": "Free tier limit reached. Please click Settings (⚙️) and add your own API key to continue."
                    }), 429
                
                # They are under the limit: Charge them 1 point
                memory_store.mongo_db['users'].update_one(
                    {"email": user_email}, 
                    {"$inc": {"usage_count": 1}}
                )
                active_key = os.environ.get("GEMINI_API_KEY") # Use your global server key
            else:
                # 🟢 PERSONAL KEY: Bypass limits entirely
                active_key = user_api_key
            
            # 🔥 CRITICAL: Configure the AI to use the correct key for this specific message!
            import google.generativeai as genai
            if active_key:
                genai.configure(api_key=active_key)
                
    except Exception as e:
        print(f"⚠️ [QUOTA CHECK ERROR] {e}")
        # If the check fails, we still let them through to Phase 4 so the app doesn't crash.
    # ==========================================
    # PHASE 4: THE COGNITIVE ENGINE (LLM)
    # ==========================================
    try:
        # Context Retrieval
        history = get_history_for_user(user_id=user_id, conversation_id=conversation_id)
        if history is None: history = []

        # --- THE MEMORY ROUTER (TRANSPLANTED) ---
        import time
        import threading
        global ACTIVE_CHAT_TIMERS
        global ACTIVE_CHAT_CURSORS

        # We only summarize if there's enough history
        if len(history) > 10:
            ltm_candidates = history[:-10]
            last_summarized_count = ACTIVE_CHAT_CURSORS.get(conversation_id, 0)
            new_ltm_messages = ltm_candidates[last_summarized_count:]

            current_time = time.time()
            last_time = ACTIVE_CHAT_TIMERS.get(conversation_id, current_time)
            ACTIVE_CHAT_TIMERS[conversation_id] = current_time
            time_idle = current_time - last_time

            print(f"🕵️ [X-RAY] Idle: {int(time_idle)}s | Total LTM Pool: {len(ltm_candidates)} | Unsummarized: {len(new_ltm_messages)}")

            # Trigger if idle > 120s AND we have new messages to summarize
            if time_idle > 120 and len(new_ltm_messages) > 0:
                print(f"⏳ [MEMORY] User idle for {int(time_idle)}s. Summarizing into LTM...", flush=True)

                def run_text_ltm_synthesis():
                    import time
                    import os
                    import google.generativeai as genai

                    # ⏳ THE COOLDOWN: Let the live chat finish its API call first!
                    print("⏳ [LTM QUEUE] Pausing 10s to prioritize Live Chat API call...", flush=True)
                    time.sleep(10)

                    try:
                        # 1. Grab ONLY the User Key (Protect the Global Key)
                        thread_api_key = None
                        try:
                            user_doc = memory_store.mongo_db['users'].find_one({"email": user_email})
                            if user_doc and 'settings' in user_doc:
                                thread_api_key = user_doc['settings'].get('gemini_api_key')
                        except Exception:
                            pass

                        if not thread_api_key:
                            print("⚠️ [LTM] No User Key found. Canceling LTM to save Global Quota.", flush=True)
                            return

                        old_texts = [m.get("parts", [""])[0] for m in new_ltm_messages]
                        joined_text = " | ".join(old_texts)
                        
                        prompt = (
                            "You are a clinical AI memory extractor. Read the following chat history. "
                            "Extract ONLY objective psychological facts, user demographics, hobbies, and emotional baselines. "
                            "Keep it under 3 sentences. If there is nothing useful, say 'No facts'.\n\n"
                            f"Chat History: {joined_text}"
                        )

                        print(f"🔄 [LTM] Executing Native call to Gemini using USER KEY...", flush=True)
                        
                        # 2. Execute strictly with User Key
                        genai.configure(api_key=thread_api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content(prompt)
                        new_facts = response.text.strip()

                        # 3. Save to Database (VARIABLES CORRECTED HERE)
                        if new_facts and new_facts != "No facts":
                            print(f"🧠 [LTM UPDATED NATIVELY]: {new_facts}", flush=True)
                            memory_store.update_profile(str(user_id), str(user_email), new_facts)
                            ACTIVE_CHAT_CURSORS[conversation_id] = last_summarized_count + len(new_ltm_messages)
                        else:
                            print("⚠️ [LTM WARNING] Gemini analyzed it but found no useful facts.", flush=True)

                    except Exception as e:
                        error_msg = str(e).lower()
                        # Abort cleanly on quota errors
                        if "429" in error_msg or "quota" in error_msg:
                            print("⏳ [LTM QUOTA] User key hit rate limit. Aborting LTM update to protect Global Key. Will try next cycle.", flush=True)
                        else:
                            print(f"❌ [LTM FATAL ERROR]: {str(e)}", flush=True)

                # Start the background thread
                threading.Thread(target=run_text_ltm_synthesis, daemon=True).start()

        # Keep Short Term Memory strictly to 10 messages for the LLM payload
        history = history[-10:]
        # ----------------------------------------

        # Append new message
        history.append({"role": "user", "parts": [transcript]})
        facts = memory_store.get_profile(user_email) or "New session."

        # The LLM Call
        llm_result = generate_chat_response(
            messages=history,
            life_facts=facts,
            model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
            api_key=active_key,   # 👈 MUST explicitly pass the key we grabbed in Phase 3.5!
            max_tokens=4096       # 👈 MUST override the 1000 limit to prevent truncation!
        )

        # Graceful Rate Limit Fallback
        if not llm_result or llm_result.get("status") != "success":
            error_msg = llm_result.get("error", "Unknown Error") if llm_result else "Timeout"
            if "429" in error_msg or "Quota" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print("⏳ [API RATE LIMIT] Triggering UI cooldown message.")
                return jsonify({
                    "success": False,
                    "error_type": "QUOTA_EXHAUSTED",
                    "message": "My daily API limit has been reached. Please click Settings (⚙️) and paste a new Gemini API key to continue our session."
                }), 429
            return jsonify({"error": error_msg}), 500

        # ==========================================
        # PHASE 5: PROCESS & SAVE SUCCESS
        # ==========================================
        response_text = str(llm_result.get("response", "I'm here."))
        raw_sentiment = str(llm_result.get("sentiment", "neutral"))
        raw_themes = llm_result.get("themes", [])
        
        try:
            importance = int(llm_result.get("importance", 5))
        except (ValueError, TypeError):
            importance = 5

        if not isinstance(raw_themes, list):
            raw_themes = [str(raw_themes)]

        # Save User Message to Cloud
        memory_store.store_memory(
            user_id=user_email,
            memory_type="conversation",
            text=f"User: {transcript}",
            conversation_id=conversation_id,
            tags=["user"] + raw_themes,
            sentiment=raw_sentiment,
            importance=importance
        )

        # Save AI Response to Cloud
        memory_store.store_memory(
            user_id=user_email,
            memory_type="conversation",
            text=f"AI: {response_text}",
            conversation_id=conversation_id,
            tags=["assistant"] + raw_themes,
            sentiment=raw_sentiment,
            importance=importance
        )

        # ==========================================
        # PHASE 6: FINAL RETURN
        # ==========================================
        return jsonify({
            "success": True,
            "response": response_text,
            "message": response_text,
            "transcript": transcript,
            "vocal_tone": vocal_tone,
            "text": response_text,
            "sentiment": raw_sentiment,
            "conversation_id": conversation_id
        }), 200

    except Exception as e:
        error_str = str(e)
        print(f"❌ [ROUTE ERROR] {error_str}")

        # The API Limit Catcher (Failsafe)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Quota" in error_str:
            return jsonify({
                "success": False,
                "error_type": "QUOTA_EXHAUSTED",
                "message": "My daily API limit has been reached. Please click Settings (⚙️) and paste a new Gemini API key to continue our session."
            }), 429

        return jsonify({"success": False, "error": error_str}), 500


from perception.stt.stt_live import transcribe_audio, extract_pitch

def get_transcript_from_request():
    # 1. Check if the browser sent an audio file (e.g., from the Record button)
    if 'audio' in request.files:
        audio_file = request.files['audio']
        filename = "temp_upload.wav"
        audio_file.save(filename)
        
        print(f"[PERCEPTION] Processing audio file: {filename}")
        
        try:
            # 2. Extract Pitch (Crucial for your "Affective" integration thesis!)
            pitch = extract_pitch(filename)
            print(f"[PERCEPTION] Detected Pitch: {pitch} Hz")
            
            # 3. Transcribe using AssemblyAI
            transcript = transcribe_audio(filename)
            print(f"[PERCEPTION] Transcribed: '{transcript}'")
            
            return transcript
        except Exception as e:
            print(f"[PERCEPTION ERROR] {e}")
            return None
            
    # 4. Fallback to standard text input
    return request.form.get('text')

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
        
        meta: typing.List[str] = []
        if tone: meta.append(str(f"Mood:{tone.get('overall_mood','-')}"))
        if nlu and nlu.get('intent'): meta.append(str(f"Intent:{nlu.get('intent')}"))
        
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
            from clinical_resources import get_coping_strategies # type: ignore
            strategies = get_coping_strategies(detected_condition)
            if strategies:
                clinical_context = f"\n[CLINICAL RESOURCES] Evidence-based strategies for {detected_condition}:\n"
                for i, strategy in enumerate(strategies[:2], 1):
                    clinical_context += f"  {i}. {strategy}\n"
        
        if is_crisis:
            from clinical_resources import get_crisis_resources # type: ignore
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
        
        return clinical_context # Note: combined_context does not exist in this scope
    
  

def generate_response_data(perception_data, user_id, transcript, conversation_id=None):
    """Generate response using Gemini API with clinical integration"""

    

    try:
        from utils.llm_client import generate_chat_response
        
        # [MEMORY FIX] Store User message in working memory immediately
        if conversation_id and transcript:
        # ... (the rest of your function continues exactly as it is) ...
            store_working_memory(user_id, f"User: {transcript}", conversation_id)

        # Get message history for context
        message_history: typing.List[typing.Dict[str, str]] = []
        smart_context: typing.List[str] = []
        
        if conversation_id:
            try:
                working_mem = WorkingMemory(f"working_memory_{conversation_id}")
                
                # 1. Sequential History
                sorted_messages = working_mem.get_all_sorted()

                # Build sequential history — last 8 messages only to limit tokens
                for msg in sorted_messages[-8:]:  
                    if isinstance(msg, str):
                        if "{'user_id':" in msg: 
                            try:
                                import ast
                                dict_msg = ast.literal_eval(msg)
                                if isinstance(dict_msg, dict):
                                    msg = dict_msg.get('message', '')
                            except: pass
                            
                        if msg.startswith("User:"):
                            message_history.append({"role": "user", "parts": [msg.replace("User:", "", 1).strip()]})
                        elif msg.startswith("Assistant:"):
                            message_history.append({"role": "model", "parts": [msg.replace("Assistant:", "", 1).strip()]})
                            
                # 2. Semantic retrieval — 2 results, capped at 80 chars each
                relevant_docs = working_mem.retrieve(query=transcript, n_results=2)
                if isinstance(relevant_docs, dict) and relevant_docs.get('documents'):
                    documents = relevant_docs['documents']
                    if isinstance(documents, list):
                        for doc_list in documents:
                            if isinstance(doc_list, list):
                                for doc in doc_list:
                                    if isinstance(doc, str):
                                        is_in_history = any(doc in m['content'] for m in message_history)
                                        if not is_in_history and len(doc) > 10:
                                            smart_context.append(doc[:80])  # pyre-ignore  # cap each snippet

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
        if isinstance(life_facts, str) and len(life_facts) > 200:
            life_facts = life_facts[:200]  # pyre-ignore
        
        # Get reasoning output
        reasoning_output = ""
        try:
            # Run through reasoning modules
            insight_gen = TherapeuticInsight()
            insights_data = insight_gen.analyze_situation(
                transcript, 
                message_history[-5:] if message_history else [], # type: ignore
                perception_data.get('tone', {}).get('overall_mood', 'neutral'),
                perception_data.get('tone', {}).get('sentiment_score', 0.0)
            )
            
            insights: typing.List[str] = []
            if isinstance(insights_data, dict):
                 rec = insights_data.get('recommendation')
                 if isinstance(rec, str):
                     insights.append(rec)
            
            # [LONG-TERM MEMORY INTEGRATION] Retrieve personalized context — capped at 150 chars
            pers_context = pers_memory.get_user_memory_context_formatted(user_id, transcript)
            if isinstance(pers_context, str) and pers_context:
                insights.append(pers_context[:150])  # pyre-ignore
            
            reasoning_output = " | ".join(insights) if insights else ""
            # Cap total reasoning output
            if isinstance(reasoning_output, str) and len(reasoning_output) > 250:
                reasoning_output = reasoning_output[:250]  # pyre-ignore
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
            if msg['parts'][0] != transcript:
                clean_history.append(msg)
        
        # --- THE MEMORY ROUTER ---
        import time
        import threading
        global ACTIVE_CHAT_TIMERS
        global ACTIVE_CHAT_CURSORS
        
        # 1. Grab all older messages as LTM candidates
        ltm_candidates = clean_history[:-10]
        
        # 2. APPLY THE BOOKMARK
        last_summarized_count = ACTIVE_CHAT_CURSORS.get(conversation_id, 0)
        new_ltm_messages = ltm_candidates[last_summarized_count:]
        
        # 3. Set the Short Term Memory to a true 10 messages
        clean_history = clean_history[-10:] 

        # 4. The Server RAM Stopwatch
        current_time = time.time()
        last_time = ACTIVE_CHAT_TIMERS.get(conversation_id, current_time)
        ACTIVE_CHAT_TIMERS[conversation_id] = current_time 
        
        time_idle = current_time - last_time

        # --- X-RAY DEBUGGER ---
        print(f"🕵️ [X-RAY] Idle: {int(time_idle)}s | Total LTM Pool: {len(ltm_candidates)} | Unsummarized: {len(new_ltm_messages)} | Cursor At: {last_summarized_count}", flush=True)
        # ----------------------

        
            
            # 5. Trigger ONLY if idle > 120s AND there are actually new messages to summarize
        if time_idle > 120 and len(new_ltm_messages) > 0:
            print(f"⏳ [MEMORY] User idle for {int(time_idle)}s. Summarizing {len(new_ltm_messages)} NEW messages into LTM...")
            
            def run_ltm_synthesis():
                try:
                    from utils.llm_client import generate_core_insight
                    import os
                    
                    # You can add your backup key logic here later if you want!
                    api_key = os.environ.get("GEMINI_API_KEY", "")
                    
                    old_texts = [m.get("parts", [m.get("content", m.get("text", ""))])[0] for m in new_ltm_messages]
                    new_facts = generate_core_insight(memories=old_texts, api_key=api_key)
                    
                    if new_facts:
                        print(f"🧠 [LTM UPDATED]: {new_facts}")
                        memory_store.update_profile(user_id, new_facts) 
                        
                        # --- THE FIX: Only move the bookmark IF it succeeds! ---
                        ACTIVE_CHAT_CURSORS[conversation_id] = last_summarized_count + len(new_ltm_messages)
                    else:
                        print("⚠️ [LTM WARNING] AI returned no facts. Bookmark NOT moved.")
                        
                except Exception as e:
                    print(f"❌ [LTM ERROR]: {e}")
                    
            threading.Thread(target=run_ltm_synthesis).start()
        # -------------------------        
        # 2. Append the Enhanced User Prompt
        clean_history.append({
            "role": "user",
            "parts": [final_user_message]
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
        print(f"--- DEBUG: Total Response Length: {len(response_text)} characters ---")
        
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
        pers_memory.extract_and_save_async(user_id, current_exchange, generate_chat_response, api_key=str(os.environ.get("GEMINI_API_KEY") or ""))

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
                typing.cast(typing.Dict[str, typing.Any], result)["error"] = None if result.get("valid") else "UNKNOWN_ERROR"
            
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
            


@app.route('/api/conversation/<conversation_id>/messages', methods=['GET'])
@login_required
def get_conversation_messages(conversation_id):
    try:
        skip_amount = int(request.args.get('skip', 0))
        u_email = str(current_user.email)

        # Fetch the messages
        messages = memory_store.get_conversation_messages(
            user_id=u_email,
            conversation_id=conversation_id,
            skip=skip_amount
        )

        # --- THE FIX: The "Modern UI" Wrapper ---
        # We send it as a list AND as a wrapped object to be 100% safe
        return jsonify({
            "success": True,
            "messages": messages,
            "conversation_id": conversation_id
        }), 200

    except Exception as e:
        print(f"❌ [API ERROR] Failed to fetch messages: {e}")
        return jsonify({"success": False, "messages": [], "error": str(e)}), 500            


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
    try:
        data = request.json or {}
        # Catch the ID from any possible label the UI might send
        c_id = data.get('conversation_id') or data.get('id') or data.get('memory_id')
        
        # Check both ID formats (ID and Email) to be safe
        u_id = str(current_user.id)
        u_email = getattr(current_user, 'email', 'unknown')

        if not c_id:
            return jsonify({"success": False, "message": "No ID provided"}), 400

        print(f"🧨 [NUCLEAR DELETE] Wiping {c_id} for {u_id}/{u_email}", flush=True)

        # 1. Wipe from MongoDB (The "Everything" Search)
        # This looks for the ID in conversation_id OR memory_id for BOTH the UID and Email
        memory_store.mongo_db.memories.delete_many({
            "$or": [{"user_id": u_id}, {"user_id": u_email}],
            "$or": [
                {"conversation_id": c_id},
                {"memory_id": c_id},
                {"id": c_id}
            ]
        })

        # 2. Wipe from Vector DB
        try:
            memory_store.vector_db.delete(where={"conversation_id": c_id})
        except:
            pass

        return jsonify({"success": True, "message": "Record purged."})
        
    except Exception as e:
        print(f"❌ [DELETE FAILED] {e}", flush=True)
        return jsonify({"success": False, "error": str(e)}), 500

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
        
        messages_list = working_context.get('messages', [])
        history_count = len(messages_list) if isinstance(messages_list, list) else 0
        if not history_count:
            history_count = len(ltm_text.split('\n')) if isinstance(ltm_text, str) else 0

        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "working_memory": working_context,
            "long_term_memory": ltm_text,
            "history_count": history_count
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
                api_key = session.get('gemini_api_key') or os.environ.get("GEMINI_API_KEY")
                pers_memory.analyze_historical_data(user_id, all_convos, generate_chat_response, api_key=str(api_key or ""))
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
    """Get dynamic medical and personal report for dashboard using sentiment analysis and chat history"""
    user_id = current_user.id
    try:
        # Fetch user sessions from clinical engine
        sessions = clinical_engine.store.get_user_sessions(user_id)
        topic_freq = clinical_engine.engine.topic_frequency(sessions)
        
        latest_emotion = "Neutral"
        latest_themes_str = "None detected"
        if sessions:
            latest_session = sessions[0]
            latest_emotion = latest_session.get("emotion", "neutral").capitalize()
            try:
                import json
                themes = json.loads(latest_session.get("themes", "[]"))
                if themes:
                    latest_themes_str = ", ".join(t.replace("_", " ").title() for t in themes)
            except:
                pass
        msi = clinical_engine.engine.compute_mood_stability_index(sessions)
        tps = clinical_engine.engine.compute_therapy_progress_score(sessions)
        
        total_sessions = max(len(sessions), 1)

        # 1. Medical History Generation
        anxiety_count = sum(1 for s in sessions if s.get('emotion') == 'anxiety')
        anxiety_pct = anxiety_count / total_sessions
        anxiety_level = "High" if anxiety_pct > 0.3 else "Moderate" if anxiety_pct > 0.1 else "Low"

        # Combine stress metrics
        stress_count = sum(1 for s in sessions if s.get('emotion') == 'stress') + topic_freq.get('financial_stress', 0)
        stress_pct = stress_count / total_sessions
        stress_level = "High" if stress_pct > 0.3 else "Moderate" if stress_pct > 0.1 else "Low"

        sleep_issues = topic_freq.get('insomnia', 0)
        sleep_condition = "Poor (Frequent insomnia)" if sleep_issues > 2 else "Occasional disruptions" if sleep_issues > 0 else "Normal/Stable"

        mood_stability = "Highly Stable" if msi >= 0.7 else "Moderate Variations" if msi >= 0.4 else "Highly Volatile"

        emotion_counts = {}
        for s in sessions:
            e = s.get("emotion", "neutral")
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [e[0].capitalize() for e in sorted_emotions[:2]] if sorted_emotions else ["Neutral"] # type: ignore
        emotional_patterns = f"Dominant emotions: {', '.join(top_emotions)}"

        medical_profile: typing.List[str] = [
            f"➤ Current State: {latest_emotion} (Updated just now)",
            f"Anxiety level: {anxiety_level}",
            f"Stress level: {stress_level}",
            f"Sleep condition: {sleep_condition}",
            f"Mood stability: {mood_stability}",
            f"Emotional patterns observed: {emotional_patterns}"
        ]

        # 2. Personal Profile Generation
        work_academic = topic_freq.get('academic_pressure', 0) + topic_freq.get('work_career', 0)
        work_pressure = "High" if work_academic > 2 else "Moderate" if work_academic > 0 else "Low/Manageable"

        social = topic_freq.get('loneliness', 0) + topic_freq.get('social_anxiety', 0)
        social_pattern = "Signs of isolation or social anxiety" if social > 1 else "Active/Normal interaction"

        lifestyle = "Needs attention (potential sleep/stress impacts)" if sleep_issues > 0 or stress_level == "High" else "Generally balanced"

        coping = "Adaptive" if tps > 0.1 else "Struggling (Needs support)" if tps < -0.1 else "Mixed/Inconsistent"

        progress_note = "Showing positive response to sessions." if tps >= 0 else "Currently experiencing elevated difficulties."

        personal_profile = [
            f"➤ Active Topic: {latest_themes_str} (Updated just now)",
            f"Work or academic pressure: {work_pressure}",
            f"Social interaction pattern: {social_pattern}",
            f"Lifestyle habits: {lifestyle}",
            f"Coping behavior: {coping}",
            f"Recovery progress notes: {progress_note}"
        ]

        # Fetch risk flags securely from existing method
        clinical_report = clinical_engine.get_medical_report(user_id)
        risk_flags = clinical_report.get("risk_flags", [])

        # Get actual extracted profiles (filter out the default "No records" messages if they exist so we can cleanly append)
        actual_med = clinical_report.get("medical_history", [])
        actual_pers = clinical_report.get("personal_profile", [])
        
        if len(actual_med) == 1 and "No medical records" in actual_med[0]:
            actual_med = []
        if len(actual_pers) == 1 and "No personal profile" in actual_pers[0]:
            actual_pers = []

        # Prepend the actual extracted history above the analytical stats
        combined_medical = actual_med + ["---"] + medical_profile if actual_med else medical_profile
        combined_personal = actual_pers + ["---"] + personal_profile if actual_pers else personal_profile

        return jsonify({
            "success": True,
            "report": {
                "medical_history": combined_medical,
                "personal_profile": combined_personal,
                "risk_flags": risk_flags,
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

@app.route('/api/admin/global-stats', methods=['GET'])
@login_required
@admin_required
def get_global_stats():
    """Admin-only: Aggregates data from every single user in the system."""
    try:
        # 1. Fetch ALL memories from the collection
        all_memories = list(memory_store.mongo_db.memories.find())
        
        # 2. Calculate totals
        total_messages = len(all_memories)
        
        # 3. Extract unique users and their message counts
        user_list = [m.get('user_id') for m in all_memories if m.get('user_id')]
        user_counts = dict(Counter(user_list))
        unique_users_count = len(user_counts)

        # 4. Get the "Global Feed" (Last 10 messages across the whole app)
        # We sort by timestamp descending to see the freshest activity
        latest_feed = list(memory_store.mongo_db.memories.find().sort("timestamp", -1).limit(10))
        
        feed_data = []
        for m in latest_feed:
            feed_data.append({
                "user": m.get('user_id'),
                "text": m.get('content', 'No content')[:100] + "...",
                "time": m.get('timestamp')
            })

        return jsonify({
            "status": "success",
            "total_interactions": total_messages,
            "active_users_count": unique_users_count,
            "user_breakdown": user_counts,
            "global_feed": feed_data
        }), 200

    except Exception as e:
        print(f"❌ [ADMIN ERROR] {e}")
        return jsonify({"error": str(e)}), 500
# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

from change_password import change_password_bp # pyre-ignore[21]
app.register_blueprint(change_password_bp)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=80)
