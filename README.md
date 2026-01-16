# AGI-119

This is a Flask application that provides a personalized AI therapist with user authentication, per-user memory storage, and LLM-powered responses.

## Features

- User authentication (signup/login)
- Per-user memory: profiles, episodic memories, conversation logs
- Vector search for memory retrieval
- Dynamic prompt building with token budgeting
- LLM integration (OpenAI GPT-3.5-turbo)
- Safety guardrails for crisis detection
- Voice and text input support

## Setup

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys:**

   - AssemblyAI: Set in config.py or directly in app.py
   - OpenAI: Set openai.api_key in app.py (for embeddings)
   - SambaNova: API key is hardcoded for demo

3. **Run the Application:**

   ```bash
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:5000`.

## Endpoints

- `/login`: User login
- `/signup`: User registration
- `/logout`: User logout
- `/`: Main chat interface (requires login)
- `/start_conversation`: Start conversation (POST)
- `/analyze`: Analyze input and get response (POST, requires login)
- `/voice-chat`: Voice input analysis (POST)
- `/memory/retrieve`: Retrieve user memories (GET)
- `/memory/store`: Store memory (POST)
- `/profile`: Get/Patch user profile (GET/PATCH)

## Architecture

- **Memory Store**: ChromaDB with OpenAI embeddings for all memories (profiles, episodic, conversations)
- **Prompt Builder**: Dynamic prompt composition with token limits
- **Safety**: Keyword-based risk detection
- **LLM**: SambaNova Meta-Llama-3.3-70B-Instruct for response generation
- **Auth**: Flask-Login with in-memory user store

## Demo Scenarios

1. Normal conversation: User chats about daily life.
2. Memory-dependent: References past conversations.
3. Crisis risk: Detects harmful keywords and responds safely.