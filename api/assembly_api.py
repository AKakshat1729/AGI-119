import assembly_api as aai
from fastapi import FastAPI, UploadFile, File
from core.agi_agent import AGI119Agent
from core.emotion_detector import detect_emotion
import os

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

app = FastAPI()
agent = AGI119Agent()

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    transcript = aai.Transcriber().transcribe(audio_path)
    text = transcript.text

    emotion = detect_emotion(text)
    response = agent.process_input(text, emotion)

    return {
        "text": text,
        "emotion": emotion,
        "response": response
    }
