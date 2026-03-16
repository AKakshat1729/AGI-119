import os
import sys
import time
import google.generativeai as genai
from flask import session
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# --- Pylance Fix ---
# This stops VS Code from complaining that functions aren't exported
genai_client: Any = genai 

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

def generate_chat_response(messages: Optional[List[Dict]] = None, model: Optional[str] = None, max_tokens: int = 1000, api_key: Optional[str] = None, life_facts: str = "") -> Dict:
    """
    Sends a chat request using Google Gemini API exclusively.
    This is the ONLY LLM endpoint - no other models are used.
    """
    global total_requests_used
    
    # Force Gemini regardless of any parameters
    return _generate_gemini_response(messages, model, max_tokens, api_key, life_facts)

def _generate_gemini_response(messages: Optional[List[Dict]] = None, model: Optional[str] = None, max_tokens: int = 1000, api_key: Optional[str] = None, life_facts: str = "") -> Dict:
    """Generate response using Google Generative AI (Gemini)"""
    global total_requests_used
    messages = messages or []
    
    try:
        # Re-load environment to pick up manual .env edits
        load_dotenv()
        
        # 1. Get API Key safely
        raw_key = api_key or os.environ.get("GEMINI_API_KEY") or session.get('gemini_api_key')
        api_key_str = str(raw_key or "")
        
        if not api_key_str:
            print("[ERROR] Missing GEMINI_API_KEY")
            return {
                "status": "error",
                "response": "I am currently offline (API Key missing). Please configure your API key in settings.",
                "error": "Missing API key"
            }

        # Debug log (safe)
        display_key = f"{api_key_str[:4]}...{api_key_str[-4:]}" if len(api_key_str) > 8 else "****"
        print(f"[LLM] Using API Key: {display_key}")

        clean_key = api_key_str.strip().replace('"', '').replace("'", "")

        # 2. Initialize Gemini API
        genai_client.configure(api_key=clean_key)
        
        # Use gemini-1.5-flash by default (Much better for free tier quota)
        model_name = str(model or os.environ.get("LLM_MODEL") or "gemini-1.5-flash")
        
        # Inject life facts into system prompt
        current_system_prompt = CLINICAL_SYSTEM_PROMPT.format(life_facts=life_facts if life_facts else "No prior history available.")
        
        model_obj = genai_client.GenerativeModel(
            model_name=model_name,
            system_instruction=current_system_prompt
        )

        # 3. Build conversation history — cap at last 10 messages to limit token usage
        conversation_messages = []
        for msg in messages[-10:]:  
            role = str(msg.get('role', 'user'))
            content = str(msg.get('content', ''))
            
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
        history_payload = conversation_messages[:-1] if conversation_messages else []
        last_message = conversation_messages[-1]["parts"][0] if conversation_messages else "Hello"
        
        chat = model_obj.start_chat(history=history_payload)
        
        response = chat.send_message(
            last_message,
            generation_config=genai_client.types.GenerationConfig(
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
        
        # ── Quota / rate-limit: auto-switch and retry logic ──
        if "quota" in error_msg.lower() or "429" in error_msg or "rate" in error_msg.lower():
            print("[LLM] Quota limit hit. Sleeping for 15 seconds to let API breathe...")
            time.sleep(15)  # The magic fix for rapid API calls
            
            if _model_switcher_available:
                print("[LLM] Attempting automatic model switch...")
                new_model = find_working_model(api_key=clean_key)
                if new_model:
                    print(f"[LLM] Retrying with model: {new_model}")
                    try:
                        retry_model_obj = genai_client.GenerativeModel(
                            model_name=new_model,
                            system_instruction=current_system_prompt
                        )
                        retry_history = conversation_messages[:-1] if conversation_messages else []
                        retry_last = conversation_messages[-1]["parts"][0] if conversation_messages else "Hello"
                        retry_chat = retry_model_obj.start_chat(history=retry_history)
                        retry_resp = retry_chat.send_message(
                            retry_last,
                            generation_config=genai_client.types.GenerationConfig(
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
                            "response": "Quota exceeded on all models. Please wait 60 seconds before sending another message.",
                            "error": "All Models Quota Exceeded"
                        }
            else:
                return {
                    "status": "error",
                    "response": "I am thinking a bit too fast! Please give me 30 seconds before your next message.",
                    "error": "Quota Exceeded — No Fallback"
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
            "response": "I'm having trouble connecting to my cognitive core. Please try again.",
            "error": error_msg
        }

def validate_gemini_api_key(api_key: Optional[str] = None) -> Dict:
    """Validates the Gemini API Key by attempting a simple generation"""
    try:
        clean_key = str(api_key or "").strip().replace('"', '').replace("'", "")
        if not clean_key:
            raise ValueError("Empty API Key provided")
            
        genai_client.configure(api_key=clean_key)
        
        # Try a simple generation to validate using the stable 1.5 model
        model = genai_client.GenerativeModel('gemini-1.5-flash')
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
        elif "quota" in error_msg.lower() or "429" in error_msg:
            return {
                "valid": False,
                "message": "API quota exceeded",
                "error": "QUOTA_EXCEEDED"
            }
        else:
            return {
                "valid": False,
                "message": f"Validation error: {error_msg}",
                "error": "VALIDATION_ERROR"
            }

def generate_core_insight(memories: Optional[List[str]] = None, api_key: Optional[str] = None) -> str:
    """Generates an ultra-concise (max 20 tokens) summary of user life facts from memories"""
    try:
        memories = memories or []
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
            return str(res.get("response", "")).strip()
        return ""
    except Exception as e:
        print(f"Error generating core insight: {e}")
        return ""