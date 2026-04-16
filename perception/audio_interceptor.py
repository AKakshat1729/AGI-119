import os
import json
from google import genai
from google.genai import types

def transcribe_audio_file(audio_file_obj) -> dict:
    """
    AGI PERCEPTION LOBE: 
    Extracts both the spoken text AND the emotional vocal tone.
    Returns a dictionary: {"transcript": text, "tone": emotion}
    """
    if not audio_file_obj or audio_file_obj.filename == '':
        return {"transcript": "", "tone": ""}

    result = {"transcript": "", "tone": ""}
    
    try:
        global_key = os.environ.get("GEMINI_API_KEY")
        if not global_key:
            raise ValueError("Global GEMINI_API_KEY missing.")
            
        client = genai.Client(api_key=global_key)

        print("🎙️ [PERCEPTION] Analyzing vocal tone and transcribing...", flush=True)
        
        audio_bytes = audio_file_obj.read()
        
        # --- THE MULTIMODAL PERCEPTION PROMPT ---
        prompt = """You are an expert clinical auditory system. 
        Listen to this audio and return a strict JSON object with exactly two keys:
        1. "transcript": The exact words spoken (in Hinglish/English).
        2. "tone": A 1-to-3 word clinical description of the speaker's emotional vocal tone (e.g., 'Anxious and shaky', 'Calm and steady', 'Aggressive', 'Crying', 'Flat/Apathetic', 'Neutral').
        Return ONLY valid JSON. Do not include markdown tags.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"  # Force JSON output!
            )
        )
        
        # Parse the JSON response
        try:
            parsed_data = json.loads(response.text.strip())
            result["transcript"] = parsed_data.get("transcript", "")
            result["tone"] = parsed_data.get("tone", "")
            print(f"✅ [PERCEPTION] Heard: '{result['transcript']}' | Tone: '{result['tone']}'", flush=True)
        except json.JSONDecodeError:
            # Fallback if the model breaks the JSON rule
            print("⚠️ [PERCEPTION] JSON parse failed, falling back to raw text.", flush=True)
            result["transcript"] = response.text.strip()

    except Exception as e:
        print(f"❌ [PERCEPTION FATAL ERROR]: {e}", flush=True)

    return result
