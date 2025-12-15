from .stt.stt_live import record_audio, save_wav, transcribe_audio
from .tone.tone_sentiment_live import analyze_tone
from .nlu.nlu_live import nlu_process

class PerceptionModule:
    def __init__(self):
        pass

    def process_audio(self, duration=5):
        # Record audio
        audio_data = record_audio(duration)
        filename = "temp_audio.wav"
        save_wav(audio_data, filename)
        # Transcribe
        text = transcribe_audio(filename)
        return text

    def process_text(self, text):
        # Analyze tone
        tone = analyze_tone(text)
        # NLU process
        nlu_output = nlu_process(text, tone)
        return nlu_output
