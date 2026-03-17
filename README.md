
# AGI Therapist (AGI-119)

A responsive, web-based AI therapist cognitive architecture that combines speech-to-text analysis, natural language understanding, emotional reasoning, and affective dual-memory integration to provide therapeutic conversations.

## 🌟 Features

- **Multimodal Input**: Users can type messages or record audio for analysis.
- **Affective Perception**: Analyzes emotional tone and sentiment from user input in real-time.
- **Dual-Memory Integration**: 
  - *Working Memory* (ChromaDB) for semantic context within the session.
  - *Long-Term Memory* (MongoDB) to retain user history and build longitudinal therapeutic context.
- **Cognitive Core**: Powered by Google's Gemini 3 Flash for rapid, empathetic, and clinical-level reasoning.
- **Responsive Interface**: Clean HTML/CSS/JS chatbot interface designed for desktop and mobile.

## 🧠 Architecture

- **Frontend**: Flask-served web interface with asynchronous API calls.
- **Perception Module**: Speech-to-text (STT), NLTK-based NLU, and Librosa/Sounddevice for audio analysis.
- **Memory Module**: Dual-database approach using MongoDB (Persistence) and ChromaDB (Vector Search).
- **Controller/Reasoning Module**: Orchestrates life understanding, emotional reasoning, and ethical boundaries using the Gemini API.

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.8+
- MongoDB installed locally

### 1. Clone the repository
```bash
git clone [https://github.com/AKakshat1729/AGI-119.git](https://github.com/AKakshat1729/AGI-119.git)
cd AGI-119

```

### 2. Set up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Requirements

```bash
pip install -r requirements.txt

```

### 4. Environment Variables

Create a `.env` file in the root directory and add the following:

```text
GEMINI_API_KEY=your_google_ai_studio_key
PORT=5000
LLM_MODEL=gemini-3-flash-preview
MONGO_URI=mongodb://localhost:27017/ 

```

### 5. Run the Application

```bash
python app.py

```

Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🔮 Future Enhancements

* User authentication and multi-tenant session management.
* Advanced clinical conversation flow management.
* Multi-language support.
* Native voice synthesis (TTS) for AI responses.

```

