

# 🧠 AGI-119: Multimodal Clinical AI Therapist

### *A Modular Cognitive Architecture for Empathetic AGI Simulation*

**AGI-119** is an advanced AI therapist designed to simulate human-like cognitive processing. Built at **Integral University**, this project implements a full **Perception-Memory-Reasoning** pipeline, allowing the AI to track a user’s clinical progress, recall long-term life events, and detect mental health crises in real-time.

---

## 🏗️ Cognitive Architecture

The system follows a modular "Brain" loop to ensure responses are not just reactive, but contextually grounded in the user's history.

1. **Perception Module**:
* **STT**: Multi-format audio transcription via **AssemblyAI**.
* **Tone Analysis**: Real-time sentiment and emotional valence detection.
* **NLU**: Extraction of user intent and clinical "flags."


2. **Memory Module (ChromaDB)**:
* **Working Memory**: Stores the current session's immediate context.
* **Long-term Episodic Memory**: A vector-stored history of past interactions for longitudinal understanding.
* **Core Insights**: Periodically extracts and persists high-level facts about the user's identity and medical history.


3. **Reasoning & Clinical Intelligence**:
* **Clinical Engine**: Automatically tracks indicators for Anxiety, Depression, and Stress.
* **Safety Engine**: Built-in **Crisis Intervention Protocol** that detects self-harm indicators and triggers emergency resources.



---

## 🛠️ Technology Stack

| Component | Technology |
| --- | --- |
| **LLM Orchestration** | Gemini Pro & Meta-Llama-3.3-70B |
| **Backend Framework** | Flask (with Flask-Login & FastAPI sub-modules) |
| **Vector Database** | ChromaDB (RAG-based Memory) |
| **Primary Database** | MongoDB Atlas (Cloud) / Local JSON Fallback |
| **Speech Services** | AssemblyAI (STT) & gTTS (TTS with Hindi/English support) |
| **Data Analytics** | Pandas, Numpy, and Scikit-Learn |

---

## 🚀 Quick Start

### 1. System Prerequisites

Before installing Python packages, ensure your system has `PortAudio` installed for audio stream handling:

* **Linux:** `sudo apt install portaudio19-dev`
* **macOS:** `brew install portaudio`

### 2. Setup Environment

```bash
# Clone and enter directory
git clone https://github.com/AKakshat1729/AGI-119.git
cd AGI-119

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Configuration

Create a `.env` file in the root directory:

```env
FLASK_SECRET_KEY=your_secret_key
GEMINI_API_KEY=your_google_ai_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
MONGO_URI=mongodb+srv://your_mongo_uri

```

### 4. Run the App

```bash
python app.py

```

Access the interface at `http://localhost:5000`.

---

## 📈 Clinical Dashboard & Analytics

The integrated **Clinical Intelligence Layer** provides real-time visualizations of user health trends.

* **Health Tracking**: Keyword-based analysis of Anxiety, Depression, and Insomnia markers.
* **Session Timeline**: Visualized history of session summaries and mood scores.
* **Personalized Reports**: Merged summaries of medical history and personal profile facts extracted from chats.

---

## 📁 Project Structure

```text
AGI-119/
├── core/               # AGI Agent & Clinical Engine orchestrators
├── memory/             # ChromaDB logic (Working vs. Long-term)
├── perception/         # STT, Tone Analysis, and NLU modules
├── reasoning/          # Ethical awareness & Emotional reasoning logic
├── static/audio/       # Storage for generated TTS responses
├── templates/          # HTML interfaces (Chat, Analytics, Dashboard)
└── app.py              # Main Flask Entry Point

```

---

## 🛡️ Medical Disclaimer

**AGI-119 is a research simulation project.** It is not a substitute for professional medical advice. If you are in a crisis, please contact emergency services or the National Suicide Prevention Lifeline at **988**.

---

## 📜 License

Distributed under the MIT License.

---


