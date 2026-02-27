# main_live.py
import asyncio, threading, sys

# Import modules from subfolders
from perception.stt.stt_live import start_stt, stop_stream
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from perception.reasoning.insight import TheraputicInsight
# In main_live.py

# Initialize the analyzer once
insight_analyzer = TheraputicInsight()


def handle_text(text, pitch=None):
    tone = analyze_tone(text)
    
    # --- NEW: Reasoning Step ---
    # We need a fake history for now (we will add real memory later)
    dummy_history = [] 
    
    # Get the "Deep Insight"
    insight = insight_analyzer.analyze_situation(
        text=text,
        history=dummy_history,
        current_emotion=tone['emotions'][0], # Takes the first detected emotion
        sentiment_score=tone['sentiment']['compound_score']
    )
    
    # Print the "Reasoning" clearly
    print(f"🎵 Pitch: {pitch:.2f} Hz")
    print(f"🧠 Insight Strategy: {insight['recommended_strategy']}")
    if insight['triggers']:
        print(f"[WARNING] Triggers Found: {insight['triggers']}")
    
    # Pass everything to NLU (optional update for later)
    result = nlu_process(text, tone)
    
    print("\n🗣️ Transcript:", text)
    print("🤖 AGI Raw Data:", result)

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
