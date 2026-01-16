from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import tempfile
import os
import json
import assemblyai as aai
import openai
from openai import OpenAI
import json
import os
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
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

aai.settings.api_key = "4bedc386183f491b9d12365c4d91e1a3"
openai.api_key = "your_openai_api_key"  # For embeddings
# SambaNova setup
openai.api_base = "https://api.sambanova.ai/v1/"
openai.api_key = "587a7fba-09f4-4bb5-a0bf-7a359629d44b"

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Memory store
memory_store = ServerMemoryStore()
prompt_builder = PromptBuilder(model="Meta-Llama-3.3-70B-Instruct")

# Simple user store (persistent with JSON)
users_file = 'users.json'

class User(UserMixin):
    def __init__(self, id, name, email, password):
        self.id = id
        self.name = name
        self.email = email
        self.password = password

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

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# Initialize Global Modules
wm = WorkingMemory()
wm_logs = []
ltm_logs = []
agent = AGI119Agent()

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

        # 2. Retrieve memories
        retrieved_bundle = memory_store.retrieve_memories(user_id, transcript, top_k=5)

        # 3. Build prompt
        style_config = {"style": "medium", "therapeutic": True}
        prompt_data = prompt_builder.build_prompt(user_id, transcript, retrieved_bundle, style_config)

        # 4. Call LLM
        response = openai.ChatCompletion.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=prompt_data["messages"],
            max_tokens=500
        )
        response_text = response.choices[0].message["content"].strip()

        # 5. Store the conversation
        memory_store.store_memory(user_id, "conversation", f"User: {transcript}\nAI: {response_text}", tags=["conversation"])

        print(f"Prompt tokens: {prompt_data['token_count']}")
        print(f"Response: {response_text}")

        return response_text

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return "I'm listening. Please go on."

# --- ROUTES ---

@app.route('/')
@login_required
def index():
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

# Memory APIs
@app.route('/memory/retrieve', methods=['GET'])
@login_required
def retrieve_memory():
    user_id = current_user.id
    query = request.args.get('query', '')
    memory_type = request.args.get('type')
    top_k = int(request.args.get('top_k', 10))
    recency_days = int(request.args.get('recency_days', 0)) if request.args.get('recency_days') else None
    bundle = memory_store.retrieve_memories(user_id, query, memory_type, top_k=top_k, recency_days=recency_days)
    return jsonify(bundle)

@app.route('/memory/store', methods=['POST'])
@login_required
def store_memory():
    user_id = current_user.id
    data = request.json
    memory_type = data.get('type')
    text = data.get('text')
    tags = data.get('tags', [])
    importance = data.get('importance', 1.0)
    memory_id = memory_store.store_memory(user_id, memory_type, text, tags, importance)
    return jsonify({"memory_id": memory_id})

@app.route('/profile', methods=['GET'])
@login_required
def get_profile():
    user_id = current_user.id
    profile = memory_store.get_profile(user_id)
    return jsonify({"profile": profile})

@app.route('/profile', methods=['PATCH'])
@login_required
def update_profile():
    user_id = current_user.id
    data = request.json
    text = data.get('text')
    memory_id = memory_store.update_profile(user_id, text)
    return jsonify({"memory_id": memory_id})

@app.route('/start_conversation', methods=['POST'])
@login_required
def start_conversation():
    user_id = current_user.id
    # Simple greeting logic
    return jsonify({"message": "Hello! I'm your AI therapist. I'm here to listen. How are you?", "type": "bot"})


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    audio_file = request.files['audio']
    audio_path = f"temp_{audio_file.filename}"
    audio_file.save(audio_path)

    transcript = aai.Transcriber().transcribe(audio_path)
    text = transcript.text

    emotion = detect_emotion(text)
    response = agent.process_input(text, emotion)
    
    os.remove(audio_path)

    return jsonify({
        "text": text,
        "emotion": emotion,
        "response": response
    })


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    print("\n--- NEW REQUEST ---")
    try:
        user_id = current_user.id
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
        # ulu = UserLifeUnderstanding(user_id=user_id)
        ig = InsightGenerator(user_id=user_id)

        additional_insights = ig.generate(transcript)
        insights = {
            # 'past_connections': ulu.connect_past_present(transcript),
            # 'life_story': ulu.build_life_story(),
            'cognitive_insights': additional_insights
        }

        # 4. Memory Storage
        wm.store(json.dumps(result), str(len(wm_logs)))
        wm_logs.append(result)
        ltm.store(json.dumps(result), str(len(ltm_logs)))
        ltm_logs.append(result)

        # Store episodic memory
        episodic_text = f"User input: {transcript}. Perception: {json.dumps(result)}. Insights: {json.dumps(insights)}."
        memory_store.store_memory(user_id, "episodic", episodic_text, tags=["analysis"])

        # 5. Response
        response_text = generate_therapist_response(result, insights, tone, user_id, transcript)

        response_data = {
            "message": response_text,
            "type": "bot",
            "analysis": {"perception": result, "insights": insights}
        }
        if 'audio' in request.files:
            response_data["transcript"] = transcript

        return jsonify(response_data)

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)