
# stt/stt_live.py
import sounddevice as sd
import requests
import time
import wave
import numpy as np
import asyncio
import os
import librosa
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "4bedc386183f491b9d12365c4d91e1a3")

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
    try:
        headers = {"authorization": API_KEY}
        with open(filename, 'rb') as f:
            response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
            response.raise_for_status()
            upload_url = response.json()["upload_url"]

        transcript_request = {
            "audio_url": upload_url
        }
        response = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
        response.raise_for_status()
        transcript_id = response.json()["id"]

        while True:
            response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "completed":
                return data["text"]
            elif data["status"] == "error":
                raise Exception(data.get("error", "Transcription failed"))
            time.sleep(1)
    except requests.RequestException as e:
        raise Exception(f"Network error during transcription: {str(e)}")
    except KeyError as e:
        raise Exception(f"Unexpected response format: {str(e)}")

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
