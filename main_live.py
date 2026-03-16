import asyncio
import threading
import sys
from typing import Any

# 1. --- Pylance-Safe Slot Definitions ---
# We use 'Any' and unique internal names to avoid Redeclaration errors
_stt_func: Any = None
_tone_func: Any = None
_nlu_func: Any = None
_InsightEngine: Any = None
stop_stream: bool = False

# 2. --- Explicit Imports with Aliasing ---
try:
    from perception.stt.stt_live import start_stt as _stt_func, stop_stream
    from perception.tone.tone_sentiment_live import analyze_tone as _tone_func
    from perception.nlu.nlu_live import nlu_process as _nlu_func
    from perception.reasoning.insight import TherapeuticInsight as _InsightEngine
except ImportError as e:
    print(f"⚠️ Critical Import Error: {e}")
    
    # Fallbacks to satisfy Pylance and prevent runtime crashes
    if _tone_func is None:
        _tone_func = lambda x: {"emotions": ["neutral"], "sentiment": {"compound_score": 0}}
    if _nlu_func is None:
        _nlu_func = lambda x, y: {"status": "fallback"}
    if _InsightEngine is None:
        # Use a different name for the fallback class to avoid redeclaration
        class _FallbackInsight:
            def analyze_situation(self, **kwargs): 
                return {"recommended_strategy": "Listen & Validate", "triggers": []}
        _InsightEngine = _FallbackInsight

# 3. --- Global Initialization ---
# These are the names the rest of your logic uses
start_stt = _stt_func
analyze_tone = _tone_func
nlu_process = _nlu_func
insight_analyzer = _InsightEngine()

def handle_text(text: str, pitch: Any = None):
    """Orchestrates the Perception -> Reasoning -> NLU pipeline"""
    # perception.tone
    tone = analyze_tone(text)
    
    # reasoning.insight
    dummy_history = [] 
    emotions = tone.get('emotions', ['neutral'])
    current_emotion = emotions[0] if emotions else 'neutral'
    
    insight = insight_analyzer.analyze_situation(
        text=text,
        history=dummy_history,
        current_emotion=current_emotion,
        sentiment_score=tone.get('sentiment', {}).get('compound_score', 0)
    )
    
    # Output logic
    print(f"\n🧠 Recommended Strategy: {insight.get('recommended_strategy')}")
    if insight.get('triggers'):
        print(f"🚨 Triggers: {insight.get('triggers')}")
    
    # perception.nlu
    result = nlu_process(text, tone)
    print("🗣️ Transcript:", text)
    print("-" * 30)

def listen_for_quit():
    """Background listener for the stop command"""
    global stop_stream
    print("\n🚀 AGI Therapist Live | Press 'q' + Enter to quit...\n")
    while True:
        key = sys.stdin.readline().strip().lower()
        if key == "q":
            # Direct update to the module variable to ensure STT loop sees it
            import perception.stt.stt_live as stt
            stt.stop_stream = True
            print("🛑 Stopping transcription...")
            break

if __name__ == "__main__":
    # Start the input listener in a background thread
    threading.Thread(target=listen_for_quit, daemon=True).start()
    
    # Launch the STT event loop
    if start_stt:
        try:
            asyncio.run(start_stt(handle_text))
        except KeyboardInterrupt:
            print("\nSession Terminated.")
    else:
        print("❌ Error: STT module not loaded. Check perception/stt/stt_live.py")