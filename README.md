
# AGI Therapist (AGI-119)

A responsive, web-based AI therapist cognitive architecture that combines speech-to-text analysis, natural language understanding, emotional reasoning, and affective dual-memory integration to provide therapeutic conversations.

Tech Stack:
Cloud Infrastructure: Microsoft Azure (Linux VM, `systemd` managed)
Backend: Python (Flask/FastAPI)
AI & Machine Learning: Google Gemini API, PyTorch, Custom LSTM Neural Networks
Multimodal Input: Users can type messages or record audio for analysis.
Affective Perception: Analyzes emotional tone and sentiment from user input in real-time.
Dual-Memory Integration: 
Working Memory (ChromaDB) for semantic context within the session.
Long-Term Memory (MongoDB) to retain user history and build longitudinal therapeutic context.
Cognitive Core: Powered by Google's Gemini 3 Flash for rapid, empathetic, and clinical-level reasoning.
Responsive Interface: Clean HTML/CSS/JS chatbot interface designed for desktop and mobile.

System Architecture:

This application is built on a multi-layered reasoning engine, acting as the "nervous system" for the AI:

1. Perception Module (LSTM & Transformers): - Processes user inputs to detect complex emotional states (Joy, Sadness, Anger, Fear, Love, Surprise).
   
2. Dual-Memory RAG Engine:
Working Memory: Tracks the immediate conversational context (short-term sequential history) to maintain fluid dialogue.
Long-Term Episodic Memory: Uses local semantic embeddings to store and retrieve core user psychological profiles and past insights, significantly reducing API quota usage while maintaining a deep understanding of the user.

3. Clinical Intelligence & Safety Engine:
Dynamically analyzes sentiment scores.
Automatically injects "Crisis Intervention Logic" into the LLM's prompt stream if severe negative valence (e.g., sentiment < -0.7) or specific high-risk keywords are detected, ensuring ethical AI behavior.


"Local Development Setup"

# Prerequisites
Python 3.8+
MongoDB installed locally

# 1. Clone the repository
```bash
git clone [https://github.com/AKakshat1729/AGI-119.git](https://github.com/AKakshat1729/AGI-119.git)
cd AGI-119

```

# 2. Set up Virtual Environment

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
MONGO_URI=xyz 

```

### 5. Run the Application

```bash
python app.py

```

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hosted_on_Azure-0078D4?style=for-the-badge&logo=microsoft-azure)](https://agitherapist.app/login?next=%2F)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Affective_LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Gemini API](https://img.shields.io/badge/Google_Gemini-LLM_Engine-4285F4?style=for-the-badge&logo=google&logoColor=white)]()

> **Note:** The architectural thesis for this project has been accepted for presentation at the **CE2CT-2026 Academic Conference**.


Core Technical Solutions (Developer Notes):
Dynamic API Routing: Built robust data parsing that dynamically shifts between `multipart/form-data` (for audio) and standard JSON text inputs over a single `/analyze` endpoint.
LLM Schema Enforcement: Engineered precise payload formatting to ensure compatibility with Gemini's strict `{"parts": [content]}` schema, preventing multimodal validation crashes during memory injection.
Session Management: Built defensive UUID state tracking that handles edge cases from frontend javascript (e.g., `null` string handling).

Live Deployment
The application is actively hosted and routing live traffic on Microsoft Azure.
Try it here: [https://agitherapist.app/login?next=%2F](https://agitherapist.app/login?next=%2F)

Author
Akshat Kashyap
AI Solutions Architect | B.Tech Computer Science (CCAI - IBM)
[LinkedIn Profile](linkedin.com/in/akshat-kashyap-7868703b4)

#Future Enhancements

* User authentication and multi-tenant session management.
* Advanced clinical conversation flow management.
* Multi-language support.
* Native voice synthesis (TTS) for AI responses.

```

