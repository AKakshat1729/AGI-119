# AI Therapist Chatbot

A responsive web-based AI therapist chatbot that combines speech-to-text analysis, natural language understanding, emotional reasoning, and long-term memory to provide therapeutic conversations.

## Features

- **Text and Audio Input**: Users can type messages or record audio for analysis
- **Real-time Speech Recognition**: Converts audio to text using AssemblyAI
- **Emotional Analysis**: Analyzes tone and sentiment from user input
- **Memory Integration**: Maintains working memory and long-term memory for context
- **Responsive Design**: Works on desktop and mobile devices
- **Conversational AI**: Provides empathetic, therapeutic responses

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

- **Text Input**: Type your message in the text box and press Enter or click Send
- **Audio Input**: Hold the microphone button to record audio, release to send
- **Conversation Flow**: The AI therapist will respond with empathetic, context-aware messages

## Architecture

- **Frontend**: HTML/CSS/JavaScript chatbot interface
- **Backend**: Flask web server
- **Perception Module**: Speech-to-text (STT), tone analysis, natural language understanding (NLU)
- **Memory Module**: Working memory and long-term memory using ChromaDB
- **Reasoning Module**: User life understanding, emotional reasoning, ethical awareness

## API Endpoints

- `GET /`: Main chatbot interface
- `POST /start_conversation`: Initialize a new conversation
- `POST /analyze`: Analyze user input (text or audio) and generate response

## Future Enhancements

- User authentication and session management
- Advanced conversation flow management
- Integration with additional therapeutic techniques
- Multi-language support
- Voice synthesis for AI responses
