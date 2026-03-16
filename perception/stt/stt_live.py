
# stt/stt_live.py
import sounddevice as sd
import time
import wave
import numpy as np
import asyncio
import os
import librosa
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

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
    try:
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
    except Exception as e:
        print(f"Pitch extraction error: {e}")
        return None

def transcribe_audio(filename):
    """
    Upload audio to AssemblyAI and get transcript using SDK with multi-language support
    """
    try:
        # Configure transcription with language detection
        # This supports Hindi, English, and Hinglish (mixed)
        config = aai.TranscriptionConfig(
            language_detection=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(filename, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
            
        return transcript.text
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise e

async def start_stt(handle_text):
    global stop_stream
    while not stop_stream:
        try:
            audio = record_audio(5)
            save_wav(audio, "temp.wav")
            pitch = extract_pitch("temp.wav")
            text = transcribe_audio("temp.wav")
            if text:
                handle_text(text, pitch)
        except Exception as e:
            print(f"STT Loop error: {e}")
        await asyncio.sleep(1)
    
    # Clean up temp file
    if os.path.exists("temp.wav"):
        try:
            os.remove("temp.wav")
        except:
            pass
