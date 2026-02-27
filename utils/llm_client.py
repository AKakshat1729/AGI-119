import os
import sys
import google.generativeai as genai
from flask import session
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Auto-model switcher — imported from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from check_gemini_models import find_working_model
    _model_switcher_available = True
except ImportError:
    _model_switcher_available = False
    print("[LLM] Warning: check_gemini_models not found — auto model-switch disabled.")

total_requests_used = 0
DAILY_REQUEST_LIMIT = 10000

# Clinical System Prompt — concise version to reduce token usage
CLINICAL_SYSTEM_PROMPT = """You are a compassionate AI therapist skilled in CBT, ACT, DBT, and positive psychology.

USER CONTEXT:
{life_facts}

APPROACH:
- Validate feelings before offering advice
- Ask 1-2 insightful questions per response to promote self-discovery
- Apply evidence-based techniques (cognitive reframing, grounding, values clarification)
- Respond to Hinglish naturally; be culturally sensitive
- Use "we" language; be warm and non-judgmental
- Each reply: acknowledge → explore → intervene → encourage next step

CRISIS: If user mentions self-harm or suicidal thoughts, immediately express concern, assess severity, and share: Call 988, text HOME to 741741, or go to ER.

LIMITS: Never diagnose or prescribe. Remind user AI is not a substitute for professional care when relevant.

RECALLED CONTEXT: If the user asks about something previously shared, check context first before answering."""

def generate_chat_response(messages, model=None, max_tokens=1000, api_key=None, life_facts=""):
    """
    Sends a chat request using Google Gemini API exclusively.
    This is the ONLY LLM endpoint - no other models are used.
    """
    global total_requests_used
    
    # Force Gemini regardless of any parameters
    return _generate_gemini_response(messages, model, max_tokens, api_key, life_facts)

def _generate_gemini_response(messages: List[Dict], model: str = None, max_tokens: int = 1000, api_key: str = None, life_facts: str = "") -> Dict:
    """Generate response using Google Generative AI (Gemini)"""
    global total_requests_used
    try:
        # Re-load environment to pick up manual .env edits
        load_dotenv()
        
        # 1. Get API Key with fallback precedence
        # We prioritize the environment variable if it's set, as the user likely updated it there.
        # Fallback to session for user-specific keys if .env is missing.
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or session.get('gemini_api_key')
        
        if not api_key:
            print("[ERROR] Missing GEMINI_API_KEY")
            return {
                "status": "error",
                "response": "I am currently offline (API Key missing). Please configure your API key in settings.",
                "error": "Missing API key"
            }

        # Debug log (safe)
        display_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        print(f"[LLM] Using API Key: {display_key}")

        clean_key = api_key.strip().replace('"', '').replace("'", "")

        # 2. Initialize Gemini API
        genai.configure(api_key=clean_key)
        
        # Use gemini-2.0-flash by default (stable, widely available)
        model_name = model or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        
        # Inject life facts into system prompt
        current_system_prompt = CLINICAL_SYSTEM_PROMPT.format(life_facts=life_facts if life_facts else "No prior history available.")
        
        model_obj = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=current_system_prompt
        )

        # 3. Build conversation history — cap at last 10 messages to limit token usage
        conversation_messages = []
        for msg in messages[-10:]:  
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                conversation_messages.append({
                    "role": "user",
                    "parts": [content]
                })
            elif role in ['assistant', 'bot', 'model']:
                conversation_messages.append({
                    "role": "model",
                    "parts": [content]
                })

        # 4. Generate response using chat
        # Ensure we have at least one message and it ends with user (or we start fresh)
        # Gemini chat history shouldn't include the very last message we want to send
        
        history_payload = conversation_messages[:-1] if conversation_messages else []
        last_message = conversation_messages[-1]["parts"][0] if conversation_messages else "Hello"
        
        chat = model_obj.start_chat(history=history_payload)
        
        response = chat.send_message(
            last_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95
            )
        )

        total_requests_used += 1

        return {
            "status": "success",
            "response": response.text,
            "tokens_used": 0,
            "error": None
        }

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Gemini API Error: {error_msg}")
        
        # ── Quota / rate-limit: auto-switch to a working model and retry once ──
        if "quota" in error_msg.lower() or "429" in error_msg or "rate" in error_msg.lower():
            if _model_switcher_available:
                print("[LLM] Quota exceeded — attempting automatic model switch...")
                new_model = find_working_model(api_key=clean_key)
                if new_model:
                    print(f"[LLM] Retrying with model: {new_model}")
                    try:
                        retry_model_obj = genai.GenerativeModel(
                            model_name=new_model,
                            system_instruction=current_system_prompt
                        )
                        retry_history = conversation_messages[:-1] if conversation_messages else []
                        retry_last = conversation_messages[-1]["parts"][0] if conversation_messages else "Hello"
                        retry_chat = retry_model_obj.start_chat(history=retry_history)
                        retry_resp = retry_chat.send_message(
                            retry_last,
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=max_tokens,
                                temperature=0.7,
                                top_p=0.95
                            )
                        )
                        return {
                            "status": "success",
                            "response": retry_resp.text,
                            "tokens_used": 0,
                            "error": None,
                            "model_switched_to": new_model
                        }
                    except Exception as retry_e:
                        print(f"[LLM] Retry with {new_model} also failed: {retry_e}")
                        return {
                            "status": "error",
                            "response": f"Quota exceeded on all models. Please wait a while or add billing to your Google API project.",
                            "error": "All Models Quota Exceeded"
                        }
                else:
                    return {
                        "status": "error",
                        "response": "Quota exceeded and no working model found. Please check your API key or wait for quota reset.",
                        "error": "Quota Exceeded — No Fallback"
                    }
            return {
                "status": "error",
                "response": "API quota exceeded. Please try again later or update your key in settings.",
                "error": "Quota Exceeded"
            }

        # ── Invalid key ──
        if "API key" in error_msg or "unauthorized" in error_msg.lower() or "expired" in error_msg.lower():
            if 'gemini_api_key' in session:
                print("[LLM] Clearing invalid/expired session API key.")
                session.pop('gemini_api_key', None)
            return {
                "status": "error",
                "response": "API key expired or invalid. Please update the key in settings.",
                "error": "Invalid API Key"
            }

        return {
            "status": "error",
            "response": "I'm having trouble connecting. Please try again.",
            "error": str(e)
        }

def validate_gemini_api_key(api_key):
    """Validates the Gemini API Key by attempting a simple generation"""
    try:
        clean_key = api_key.strip().replace('"', '').replace("'", "")
        genai.configure(api_key=clean_key)
        
        # Try a simple generation to validate
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say 'API working'")
        
        if response.text:
            return {
                "valid": True,
                "message": "Gemini API key is valid!",
                "error": None
            }
        else:
            return {
                "valid": False,
                "message": "API key validation failed",
                "error": "No response"
            }
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "invalid" in error_msg.lower():
            return {
                "valid": False,
                "message": "Invalid API key format",
                "error": "INVALID_KEY"
            }
        elif "quota" in error_msg.lower():
            return {
                "valid": False,
                "message": "API quota exceeded",
                "error": "QUOTA_EXCEEDED"
            }
        else:
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}",
                "error": "VALIDATION_ERROR"
            }

def generate_core_insight(memories: List[str], api_key: str = None) -> str:
    """Generates an ultra-concise (max 20 tokens) summary of user life facts from memories"""
    try:
        if not memories:
            return ""
            
        combined_mems = "\n".join(memories[:20]) # Take top 20 memories
        
        prompt = f"""Based on these user memories, identify the single most important "core life truth" or insight about this user.
        Format it as a single sentence or a few keywords. 
        CRITICAL: Use NO MORE than 15-20 words total.
        Examples: "Struggles with career identity but finds peace in nature" or "Anxious high-achiever grieving a parent."
        
        Memories:
        {combined_mems}
        """
        
        messages = [{"role": "user", "content": prompt}]
        # Use simple generation
        res = _generate_gemini_response(messages, api_key=api_key, max_tokens=50)
        
        if res.get("status") == "success":
            return res.get("response", "").strip()
        return ""
    except Exception as e:
        print(f"Error generating core insight: {e}")
        return ""