Here is a polished, fully updated `README.md` that merges your original descriptions with all the architectural upgrades and server deployment commands we established.

You can copy this directly and replace your current README file!

---

```markdown
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

## 💻 Local Development Setup (Windows / macOS)

### Prerequisites
- Python 3.8+
- MongoDB installed locally (or an Atlas Cloud URI)

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

## 🚀 Server Deployment Guide (Ubuntu / Linux)

If deploying this architecture to a fresh cloud server (e.g., Azure, AWS, DigitalOcean), follow these steps to prepare the environment.

### 1. Install System Dependencies

Your server needs audio libraries and database engines before Python can run the app.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libportaudio2 libportaudiocpp0 portaudio19-dev ffmpeg tmux

```

### 2. Install and Start MongoDB (Long-Term Memory)

```bash
# Import key and add repository
sudo apt-get install gnupg curl -y
curl -fsSL [https://www.mongodb.org/static/pgp/server-7.0.asc](https://www.mongodb.org/static/pgp/server-7.0.asc) | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] [https://repo.mongodb.org/apt/ubuntu](https://repo.mongodb.org/apt/ubuntu) jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install and start the service
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod

```

### 3. Setup Application

```bash
git clone [https://github.com/AKakshat1729/AGI-119.git](https://github.com/AKakshat1729/AGI-119.git)
cd AGI-119
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

*(Don't forget to create your `.env` file on the server using `nano .env`!)*

### 4. Run Persistently using Tmux

To keep the AGI running after you close your SSH connection:

```bash
tmux new -s agi_server
source venv/bin/activate
python app.py

```

*To detach from the session, press `Ctrl+B`, then `D`. To return later, type `tmux attach -t agi_server`.*

---

## 🔮 Future Enhancements

* User authentication and multi-tenant session management.
* Advanced clinical conversation flow management.
* Multi-language support.
* Native voice synthesis (TTS) for AI responses.

```
