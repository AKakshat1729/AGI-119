import os
import sys
import time
import json
import random
from google import genai
from google.genai import types
from flask import session
from typing import List, Dict, Any, Optional, cast
from dotenv import load_dotenv
# --- Cognitive Architecture Setup ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global State for Resource Management
total_requests_used = 0
last_request_time = 0.0

try:
    from check_gemini_models import find_working_model
    _model_switcher_available = True
except ImportError:
    _model_switcher_available = False
    print("[LLM] Warning: check_gemini_models not found — auto model-switch disabled.")
# --- DYNAMIC MODEL CACHE ---
_CACHED_MODELS = {}

def get_dynamic_fallback_models(api_key: str, preferred_model: str) -> list:
    """Fetches live models from Google, filters for chat, and caches them to prevent latency."""
    global _CACHED_MODELS

    # 1. Return instantly if we already fetched them for this key
    if api_key in _CACHED_MODELS:
        cached_list = _CACHED_MODELS[api_key]
        if preferred_model in cached_list:
            cached_list.remove(preferred_model)
        return [preferred_model] + cached_list

    print("🔍 [LLM] Booting dynamic model radar... Fetching live list from Google.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        available_text_models = []
        for m in client.models.list():
            if "generateContent" in m.supported_actions:
                clean_name = m.name.replace("models/", "")
                
                # Filter out pure audio/image/robotics/TTS models so they don't break the text chat
                if not any(x in clean_name for x in ["image", "audio", "tts", "lyria", "robotics", "banana", "computer-use", "deep-research"]):
                    available_text_models.append(clean_name)

        # 2. Force the absolute best models to the top of the fallback line
        ideal_priority = [
            preferred_model,
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash-lite",
            "gemini-flash-latest"
        ]

        final_list = []
        # Add the priority models first (if they actually exist on this key)
        for ideal in ideal_priority:
            if ideal in available_text_models and ideal not in final_list:
                final_list.append(ideal)
                
        # Add the rest of the available text models to the bottom of the list
        for active in available_text_models:
            if active not in final_list:
                final_list.append(active)

        # 3. Save to memory so it never has to fetch again until server restart
        _CACHED_MODELS[api_key] = final_list
        print(f"✅ [LLM] Cached {len(final_list)} active text models for this API key.")
        return final_list

    except Exception as e:
        print(f"⚠️ [LLM] Failed to fetch dynamic models: {e}. Falling back to static.")
        return [preferred_model, "gemini-2.5-flash", "gemini-2.5-flash-lite"]
# --- Consolidated Clinical Prompt ---
# This forces the LLM to handle Sentiment, Themes, and Chat in a single request.
# --- Consolidated Clinical Prompt (V2: Concise & Human) ---
CLINICAL_SYSTEM_PROMPT = """You are a compassionate AI therapist skilled in CBT, ACT, DBT, and positive psychology.

USER CONTEXT :
{life_facts}

APPROACH:
- Validate feelings before offering advice
- Ask 1-2 insightful questions per response to promote self-discovery
- Apply evidence-based techniques (cognitive reframing, grounding, values clarification)
- Respond to Hinglish naturally; be culturally sensitive
- Use "we" language; be warm and non-judgmental
- Each reply: acknowledge → explore → intervene → encourage next step



CRITICAL OUTPUT INSTRUCTIONS:

CRASH WARNING: NEVER use unescaped double quotes (") inside your response string. If you need to quote something, use single quotes (') to prevent JSON parsing failures.

Your JSON MUST strictly follow this exact schema:
{{
    "response": "Your therapeutic reply to the user. Keep it conversational, empathetic, and naturally guide them to cognitive insights.",
    "sentiment": "Select the SINGLE most accurate clinical state from this list: [euthymic, somatic_anxiety, hypervigilance, cognitive_dissonance, dopamine_seeking, emotional_blunting, depressive_rumination, emotional_dysregulation, executive_dysfunction, acute_stress]",
    "themes": ["List 2 to 3 core psychological themes present in the user's input, e.g., 'dopamine_seeking', 'social_anxiety', 'stress'"]
}}

CRISIS: If self-harm is mentioned, share: Call 988 or text HOME to 741741.
"""

def generate_chat_response(messages: Optional[List[Dict]] = None, model: Optional[str] = None, max_tokens: int = 4000, api_key: Optional[str] = None, life_facts: str = "") -> Dict:
    """Main entry point for the AGI Therapist's reasoning engine."""
    return _generate_gemini_response(messages, model, max_tokens, api_key, life_facts)

def clean_json_response(raw_text: str) -> dict:
    """Attempts to salvage broken JSON if the LLM gets cut off by a token limit."""
    import re
    
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    
    try:
        # Try the easy way first
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass # It's broken. Time for surgery.
        
    # Extract whatever was in the "response" field
    match = re.search(r'"response"\s*:\s*"([^"]*)', cleaned)
    
    if match:
        salvaged_text = match.group(1)
        # Clean up trailing partial words/sentences
        last_period = salvaged_text.rfind('.')
        if last_period != -1:
            salvaged_text = salvaged_text[:last_period + 1]
            
        return {
            "response": salvaged_text,
            "sentiment": "neutral", 
            "themes": []            
        }
    
    return {"response": cleaned, "sentiment": "neutral", "themes": []}

def _generate_gemini_response(messages: Optional[List[Dict]] = None, model: Optional[str] = None, max_tokens: int = 4500, api_key: Optional[str] = None, life_facts: str = "") -> Dict:
    global total_requests_used
    global last_request_time
    from dotenv import load_dotenv
    load_dotenv()

    # 1. Identity & Key Validation (User UI Key FIRST, then Global .env Key)
    user_key = session.get('gemini_api_key')
    env_key = os.environ.get("GEMINI_API_KEY")
    
    unique_keys = []
    if user_key and str(user_key).strip():
        unique_keys.append(str(user_key).strip())
    if env_key and str(env_key).strip() and str(env_key).strip() not in unique_keys:
        unique_keys.append(str(env_key).strip())

    # IF NO KEYS EXIST AT ALL
    if not unique_keys:
        return {
            "status": "success",
            "response": "⚙️ **System Alert**: No API keys found. Please go to **System Config -> API & Model** and enter your Gemini API Key to start chatting.",
            "sentiment": "neutral",
            "themes": ["system_error"]
        }

    # 2. Build the Arsenal (Newest to most reliable)
    preferred_model = str(model or os.environ.get("LLM_MODEL") or "gemini-2.5-flash")
    active_fetch_key = unique_keys[0] if unique_keys else None
    
    if active_fetch_key:
        models_to_try = get_dynamic_fallback_models(active_fetch_key, preferred_model)
    else:
        models_to_try = [preferred_model]
    
    # 3. Build Instructions (Memory injected here)
    system_instruction = CLINICAL_SYSTEM_PROMPT.format(life_facts=life_facts if life_facts else "No prior history.")
    if life_facts and "New session" not in life_facts:
        system_instruction += f"\n\n[DATABASE CONTEXT OVERRIDE]: You have access to the user's Long Term Memory. The user's verified profile is: {life_facts}. CRITICAL INSTRUCTION: DO NOT explicitly recite, list, or blurt out these facts to the user. DO NOT say 'I see you like bananas'. Use this information INVISIBLY as background context to shape a natural, personalized therapeutic conversation."
    # 4. Prepare Message Format
    contents = []
    for msg in (messages or [])[-15:]:
        role = "user" if msg.get('role') == 'user' else "model"
        text = msg['parts'][0] if isinstance(msg.get('parts'), list) else msg.get('content', '')
        contents.append(types.Content(role=role, parts=[types.Part(text=str(text))]))

    if not contents:
        contents.append(types.Content(role="user", parts=[types.Part(text="Hello, I'm starting a new session.")]))

    last_error_msg = ""

    # --- THE CASCADE ENGINE ---
    for active_model in models_to_try:
        for current_key in unique_keys:
            try:
                key_source = "User UI Key" if current_key == user_key else "Global .env Key"
                print(f"🚀 [LLM] Trying {active_model} | {key_source}: ...{current_key[-4:]}")
                
                client = genai.Client(api_key=current_key)
                # --- TOKEN RADAR ---
                # --- TOKEN RADAR ---
                try:
                    # Temporarily merge system instructions to get an accurate total weight
                    token_contents = list(contents)
                    token_contents.insert(0, types.Content(role="user", parts=[types.Part(text=f"System: {system_instruction}")]))
                    
                    token_response = client.models.count_tokens(
                        model=active_model,
                        contents=token_contents
                    )
                    print(f"📊 [TOKEN RADAR] Payload size: {token_response.total_tokens} tokens for {active_model}")
                except Exception as e:
                    print(f"⚠️ [TOKEN RADAR] Could not fetch exact count: {str(e)[:60]}")
                # -------------------
                
                
                # ... the rest of your existing generate_content code ...
                response = client.models.generate_content(
                    model=active_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json"
                    )
                )

                # SUCCESS: Return immediately
                res_json = clean_json_response(response.text)
                final_text = res_json.get("response", "Error parsing response.")
                
                # --- THE UI QUOTA WARNING ---
                # If the user has a key, but the key that actually succeeded here was the Global Key,
                # it means their personal key failed or ran out of quota. Add the warning!
                if user_key and current_key != user_key:
                    warning_msg = "\n\n*(System Note: Your personal Gemini API key has run out of daily quota. I used the server backup key this time to keep chatting, but please update your key in Settings using a different Google account.)*"
                    final_text += warning_msg

                return {
                    "status": "success",
                    "response": final_text,
                    "sentiment": res_json.get("sentiment", "neutral"),
                    "themes": res_json.get("themes", [])
                }

            except Exception as e:
                last_error_msg = str(e)
                print(f"❌ [FAIL] {active_model} / ...{current_key[-4:]} failed. Moving to next....THE REAL ERROR IS: {str(e)}")
                import time
                time.sleep(2)
                continue 

    # --- THE EXHAUSTION POINT ---
    # Reached only if all models and all keys fail
    return {
        "status": "success", 
        "response": "⚠️ **SYSTEM EXHAUSTED**: I've tried every available model and API key, but we have hit Google's quota limits across the board. **Please go to Google AI Studio, create a new API key with a different Google account, and paste it into the System Config UI** to continue our session.",
        "sentiment": "neutral",
        "themes": ["quota_exhausted"]
    }
def validate_gemini_api_key(api_key: Optional[str] = None) -> Dict:
    try:
        clean_key = str(api_key or "").strip().replace('"', '').replace("'", "")
        client = genai.Client(api_key=clean_key)
        client.models.generate_content(model='gemini-2.5-flash', contents="Say 'OK'")
        return {"valid": True, "message": "Key Verified!", "error": None}
    except Exception as e:
        return {"valid": False, "message": "Invalid Key", "error": str(e)}

def generate_core_insight(memories: Optional[List[str]] = None, api_key: Optional[str] = None) -> str:
    try:
        memories = memories or []
        if not memories: return ""
        prompt = f"Summarize these memories into one core life truth (max 20 words): {' '.join(memories[:20])}"
        res = _generate_gemini_response([{"role": "user", "content": prompt}], api_key=api_key, max_tokens=50)
        return str(res.get("response", "")).strip() if res.get("status") == "success" else ""
    except:
        return ""

def evaluate_memory_importance(text: str) -> int:
    """
    LOCAL NLP BOUNCER: Scores text importance (1-10) using purely local rules.
    Zero API calls. Zero cost. Filters out conversational junk AND AI responses.
    """
    if not text:
        return 1

    text_lower = text.lower().strip()

    # --- 0. THE AI BLOCKER (The Hallucination Firewall) ---
    # Prevents the AGI from memorizing its own advice in the vector database
    if text_lower.startswith("ai:") or text_lower.startswith("model:") or text_lower.startswith("assistant:"):
        print("🛡️ [BOUNCER] Blocked AI response from polluting Long Term Memory.")
        return 1

    # --- 1. THE TRASH CAN (Automatic Rejects) ---
    # Instantly drop one-word answers, greetings, and short filler
    filler_words = {"hi", "hello", "ok", "okay", "yes", "no", "yeah", "yep", "nope", "thanks", "cool", "hmm", "wow", "sure", "right"}
    if text_lower in filler_words:
        return 1
        
    if len(text_lower) < 15 and not any(word in text_lower for word in ["i", "my", "me"]):
        return 2

    score = 4  # Baseline for a normal sentence

    # --- 2. THE PSYCHOLOGICAL TRIGGERS (The Gold) ---
    # Words that indicate high-value therapy data
    emotion_words = {"feel", "sad", "angry", "happy", "anxious", "depressed", "stressed", "confused", "hurt", "love", "hate", "tired", "lazy", "fear", "trauma"}
    identity_words = {"i am", "my name", "i live", "my favorite", "i enjoy", "i hate", "i love", "my goal", "i want", "phobia", "diagnosed"}
    relational_words = {"mom", "dad", "mother", "father", "brother", "sister", "friend", "partner", "wife", "husband", "boss", "family"}

    # Boost score based on content density
    if any(word in text_lower for word in emotion_words): 
        score += 2
    if any(word in text_lower for word in identity_words): 
        score += 3
    if any(word in text_lower for word in relational_words): 
        score += 2

    # --- 3. STRUCTURAL COMPLEXITY ---
    # "Because" indicates reasoning. Length indicates a story.
    if "because" in text_lower or "why" in text_lower: 
        score += 1
    if len(text_lower) > 100: 
        score += 1

    # Ensure the score stays within the 1-10 boundary
    return min(max(score, 1), 10)
