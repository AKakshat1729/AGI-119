
# stt/stt_live.py
import sounddevice as sd
import requests
import time
import wave
import numpy as np
import asyncio
import os
import librosa
from config import API_KEY   # import from root config

stop_stream = False

def record_audio(duration=5):
    """
    Record audio from microphone for given duration
    """
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return recording

def save_wav(data, filename):
    """
    Save numpy array to wav file
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())

def extract_pitch(filename):
    """
    Extract pitch (fundamental frequency) from audio file using librosa
    Returns average pitch in Hz or None if pitch not found
    """
    y, sr = librosa.load(filename, sr=16000)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)

    if pitch_values:
        avg_pitch = float(np.mean(pitch_values))
        return avg_pitch
    else:
        return None

def transcribe_audio(filename):
    """
    Upload audio to AssemblyAI and get transcript
    """
    headers = {"authorization": API_KEY}
    with open(filename, 'rb') as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    upload_url = response.json()["upload_url"]

    transcript_request = {
        "audio_url": upload_url
    }
    response = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
    transcript_id = response.json()["id"]

    while True:
        response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        data = response.json()
        if data["status"] == "completed":
            return data["text"]
        elif data["status"] == "error":
            raise Exception(data["error"])
        time.sleep(1)

async def start_stt(handle_text):
    global stop_stream
    while not stop_stream:
        audio = record_audio(5)
        save_wav(audio, "temp.wav")
        pitch = extract_pitch("temp.wav")
        text = transcribe_audio("temp.wav")
        if text:
            handle_text(text, pitch)
        await asyncio.sleep(1)
    # Clean up temp file
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
