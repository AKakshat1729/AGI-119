from flask import Flask, render_template, jsonify, request
import tempfile
import os
from perception.stt.stt_live import save_wav, transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory

# Initialize Flask application
app = Flask(__name__)

# Initialize memory modules for short-term and long-term memory
wm = WorkingMemory() # Working memory instance

# In-memory logs for display in the web interface
wm_logs = [] # Logs for working memory operations
ltm_logs = [] # Logs for long-term memory operations

# Define the route for the index page
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the route for analyzing audio input
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get user_id from the request form data, default to 'default' if not provided
        user_id = request.form.get('user_id', 'default')

        # Initialize long-term memory instance for the specific user
        ltm = LongTermMemory(user_id=user_id)

        # Check if audio file is present in the request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400

        # Save uploaded audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filename = f.name
        audio_file.save(filename)

        # Transcribe the audio file using speech-to-text
        transcript = transcribe_audio(filename)
        # Clean up the temporary audio file
        os.unlink(filename)
        # Analyze the tone of the transcribed text
        tone = analyze_tone(transcript)
        # Process the transcribed text and tone using natural language understanding
        result = nlu_process(transcript, tone)

        # Store the result in working memory
        try:
            wm.store(result, str(len(wm_logs)))
            wm_logs.append(result)
        except Exception as e:
            wm_logs.append({"error": f"WM store failed: {str(e)}"})

        # Store the result in long-term memory
        try:
            ltm.store(str(result), str(len(ltm_logs)))
            ltm_logs.append(result)
        except Exception as e:
            ltm_logs.append({"error": f"LTM store failed: {str(e)}"})

        # Return the analysis results, working memory logs, and long-term memory logs as a JSON response
        return jsonify({
            "perception": result,
            "working_memory": wm_logs,
            "long_term_memory": ltm_logs
        })
    except Exception as e:
        # Handle any exceptions and return an error message as a JSON response
        return jsonify({"error": str(e)}), 500

# Define a route to test the working memory
@app.route('/test_wm', methods=['GET'])
def test_wm():
    """
    Test the working memory by storing and retrieving a value.
    """
    wm.store({"test": "test"}, "1")
    result = wm.retrieve("test")
    wm.clear()
    return jsonify({"result": result})

# Define a route to test the long-term memory
@app.route('/test_ltm', methods=['GET'])
def test_ltm():
    """
    Test the long-term memory by storing, retrieving, and updating a value.
    """
    user_id = request.args.get('user_id', 'default')
    ltm = LongTermMemory(user_id=user_id)
    ltm.store("test", "1")
    result = ltm.retrieve("test")
    ltm.update("1", "test2")
    result2 = ltm.retrieve("test2")
    return jsonify({"result": result, "result2": result2})

if __name__ == '__main__':
    app.run(debug=True)
