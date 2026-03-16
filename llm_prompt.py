import os
from typing import Any, List, Dict
from dotenv import load_dotenv

# --- Internal Import ---
# We link this to the 'brain' of the AGI we built in your utils folder
try:
    from utils.llm_client import generate_chat_response
except ImportError:
    # If this is a standalone script, we import the generator directly
    import google.generativeai as genai
    genai_client: Any = genai

load_dotenv()

def generate_response(prompt: str) -> str:
    """
    Standardized response generator for the AGI Therapist.
    Now powered by Gemini to handle clinical context and Hinglish.
    """
    # Safety wrap: Ensure prompt is never None
    safe_prompt = str(prompt or "")
    
    # We use our consolidated wrapper to handle the Gemini API call
    # This automatically uses the GEMINI_API_KEY from your .env
    result = generate_chat_response(
        messages=[{"role": "user", "content": safe_prompt}],
        max_tokens=8192
    )
    
    # Ensure the return value is strictly a string
    if result.get("status") == "success":
        return str(result.get("response", ""))
    
    return f"Error: {result.get('error', 'Unknown response error')}"