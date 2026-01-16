from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import tempfile
import os
import json
import assemblyai as aai
import openai
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from dotenv import load_dotenv
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory
from reasoning.user_life_understanding import UserLifeUnderstanding
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
openai.api_key = os.environ.get("SAMBA_API_KEY", "587a7fba-09f4-4bb5-a0bf-7a359629d44b")

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

def generate_therapist_response(perception_result, insights, tone, user_id="default", transcript=""):
    try:
        # 1. SAFETY CHECK (Priority #1)
        if safety_engine.detect_high_risk(transcript):
            print("⚠️ HIGH RISK DETECTED - Triggering Safety Protocol")
            return safety_engine.ethical_response()

        # 2. Retrieve memories
        try:
            retrieved_bundle = memory_store.retrieve_memories(user_id, transcript, top_k=5)
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            retrieved_bundle = {"profile_summary": "", "top_memories": [], "recency_window": [], "risk_flags": []}

        # 3. Build prompt
        try:
            style_config = {"style": "medium", "therapeutic": True}
            prompt_data = prompt_builder.build_prompt(user_id, transcript, retrieved_bundle, style_config)
        except Exception as e:
            print(f"Error building prompt: {str(e)}")
            prompt_data = {"messages": [{"role": "user", "content": transcript}], "debug_prompt_text": transcript, "token_count": len(transcript)}

        # 4. Call LLM
        try:
            response = openai.ChatCompletion.create(
                model="Meta-Llama-3.3-70B-Instruct",
                messages=prompt_data["messages"],
                max_tokens=500
            )
            response_text = response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"LLM error: {str(e)}")
            response_text = "I'm having trouble connecting right now. Can you please try again?"

        # 5. Store the conversation
        try:
            memory_store.store_memory(user_id, "conversation", f"User: {transcript}\nAI: {response_text}", tags=["conversation"])
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")

        print(f"Prompt tokens: {prompt_data['token_count']}")
        print(f"Response: {response_text}")

        return response_text

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return "I'm listening. Please go on."

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
CORS(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user store (persistent with JSON)
users_file = 'users.json'

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
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                data = json.load(f)
                return {k: User(v['id'], v.get('name', v.get('username', '')), v.get('email', v.get('username', '')), v['password']) for k, v in data.items()}
    except:
        pass
    return {}

def save_users():
    data = {k: {'id': v.id, 'name': v.name, 'email': v.email, 'password': v.password} for k, v in users.items()}
    with open(users_file, 'w') as f:
        json.dump(data, f)

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
        user = next((u for u in users.values() if u.email == email and u.password == password), None)
        if user:
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if email in [u.email for u in users.values()]:
            flash('Email already exists')
        else:
            user_id = str(len(users) + 1)
            user = User(user_id, name, email, password)
            users[user_id] = user
            save_users()
            login_user(user)
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

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
        # Check if text or audio is provided
        if 'text' in request.form and request.form['text'].strip():
            transcript = request.form['text'].strip()
        elif 'audio' in request.files:
            transcript = "Audio received"
        else:
            return jsonify({"error": "No text or audio provided"}), 400

        response_data = {
            "message": "I understand. Tell me more.",
            "type": "bot",
            "analysis": {
                "perception": {"intent": "unknown"},
                "tone": {"overall_mood": "neutral"},
                "insights": {}
            }
        }
        if 'audio' in request.files:
            response_data["transcript"] = transcript

        return jsonify(response_data)
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        return jsonify({"error": "An error occurred during analysis. Please try again."}), 500

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

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
