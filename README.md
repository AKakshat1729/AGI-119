# AGI Therapist: Cognitive Architecture Project

**Project:** AGI-119 / My-Virtual-Therapist
**Type:** Advanced AI Cognitive System (BCA 3rd Year Capstone)

## 📋 Project Overview
This is not just a chatbot; it is a multi-modular **Cognitive Architecture** designed for automated therapeutic interaction. Unlike simple LLM wrappers, this system utilizes distinct modules for Perception, Memory, Reasoning, and Safety to create a "Hybrid Brain."

It features a **Unified Vector Memory** system that bridges different code modules into a single long-term storage, ensuring data consistency and context retention.

---

## 🚀 Key Technical Features

### 1. 🧠 Unified Vector Memory (ChromaDB)
- **Architecture:** Centralized Vector Database for Long-Term Memory.
- **Data Bridging:** Solved "Split Brain" issues by bridging the `core` module (teammate's code) with the `memory` module into a single persistent database.
- **Capabilities:** Semantic search, automatic timestamping, and cross-module retrieval.

### 2. 🛡️ Ethical Safety & Perception
- **Safety Gatekeeper:** A dedicated `EthicalAwarenessEngine` intercepts user input *before* processing. It detects high-risk phrases (e.g., self-harm) and overrides the system to provide immediate help.
- **Style Engine:** Analyzes sentiment polarity to dynamically suggest response tones (Gentle, Calm, Positive).

### 3. 💡 Deep Reasoning (Internal Cognition)
- **Intent Recognition:** Identifies "True Intent" (e.g., seeking support vs. seeking information) beyond simple keywords.
- **Self-Reflection:** The system logs a self-critique of its own confidence levels and understanding after every interaction.

### 4. 📝 Dynamic Prompt Engineering
- **Context Construction:** Dynamically builds prompts using User History, Current Emotion, and Safety Constraints before sending them to the response generator.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/AKakshat1729/AGI-119.git](https://github.com/AKakshat1729/AGI-119.git)
   cd AGI-119


#Create a Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate


#Install Dependencies

pip install -r requirements.txt

#How to Run the Demo
#Step 1: Verify Backend & Memory Integrity

python verify_memory.py

#Launch the Cognitive Engine (The App)

#Step 2 : python app.py

Open your browser at http://127.0.0.1:5000

Test Emotional Intelligence: Type "I am feeling stressed but I will handle it." (Observe "Neutral/Balanced" style in terminal).

Test Safety Protocol: Type "I want to kill myself." (Observe ⚠️ HIGH RISK DETECTED in terminal).


Step 3: Audit Patient History

After stopping the app (Ctrl+C)


python view_history.py

📂 Architecture Structure
frontend/: HTML/CSS/JavaScript interface.

perception/: STT (AssemblyAI) and NLU processing.

memory/: ChromaDB Vector Database logic.

reasoning/: Internal Cognition and Insight Generators.

core/: Shared utilities (Safety, Style engines).

generation/: Prompt construction and LLM interfacing.

api/: External API handlers.

🔮 Future Enhancements
User Authentication & Multi-user session management.

Voice Synthesis (TTS) for audio responses.

Advanced Cognitive Behavioral Therapy (CBT) modules.


