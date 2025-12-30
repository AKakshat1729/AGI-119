from flask import Flask, render_template, jsonify, request
import tempfile
import os
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory
from reasoning.internal_cognition import InternalCognition

app = Flask(__name__)

# Initialize memory modules
wm = WorkingMemory()
wm_logs = [] 
ltm_logs = [] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        user_id = request.form.get('user_id', 'default')
        ltm = LongTermMemory(user_id=user_id)

        # 1. PERCEPTION LAYER
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filename = f.name
        audio_file.save(filename)

        transcript = transcribe_audio(filename)
        os.unlink(filename)
        
        tone = analyze_tone(transcript)
        result = nlu_process(transcript, tone)

        # 2. COGNITION LAYER (The Brain)
        cognition = InternalCognition(user_id=user_id)
        
        # Pull values safely
        sentiment_score = tone.get('score', 0.0) 
        current_emotion = tone.get('label', 'neutral')
        history_data = ltm.get_all().get('documents') or []

        insights = cognition.analyze_situation(
            text=transcript, 
            history=history_data, 
            current_emotion=current_emotion, 
            sentiment_score=sentiment_score
        )

        # 3. MEMORY STORAGE
        try:
            wm.store(result, str(len(wm_logs)))
            wm_logs.append(result)
            
            ltm.store(str(result), str(len(ltm_logs)))
            ltm_logs.append(result)
        except Exception as e:
            print(f"Memory store error: {e}")
        
        print(f"DEBUG: Transcript -> {transcript}")
        print(f"DEBUG: Insights Object -> {insights}")
        
        # 4. FINAL RESPONSE...
        # 4. FINAL RESPONSE
        return jsonify({
            "transcript": transcript,
            "perception": result,
            "cognitive_insights": insights,  # <--- Check if this exact line exists
            "recommendation": insights.get('summary', 'No summary available'),
            "working_memory": wm_logs,
            "long_term_memory": ltm_logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)