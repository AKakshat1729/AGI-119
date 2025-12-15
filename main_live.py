# main_live.py
import asyncio, threading, sys

# Import modules from subfolders
from perception.stt.stt_live import start_stt, stop_stream
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process

def handle_text(text):
    tone = analyze_tone(text)
    result = nlu_process(text, tone)
    print("\n🗣️ Transcript:", text)
    print("🤖 AGI Response:", result)

def listen_for_quit():
    global stop_stream
    print("\nPress 'q' + Enter anytime to quit...\n")
    while True:
        key = sys.stdin.readline().strip().lower()
        if key == "q":
            stop_stream = True
            print("🛑 Stopping transcription...")
            break

if __name__ == "__main__":
    # background quit listener
    threading.Thread(target=listen_for_quit, daemon=True).start()
    # run STT loop
    asyncio.run(start_stt(handle_text))
