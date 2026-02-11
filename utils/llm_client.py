import os
from google import genai  # Uses the NEW google-genai library
from flask import session

total_requests_used = 0
DAILY_REQUEST_LIMIT = 1000

def generate_chat_response(messages, model="gemini-2.5-flash-lite", max_tokens=500, api_key=None):
    """
    Sends a chat request using the modern Google GenAI SDK.
    Forces usage of gemini-2.5-flash to avoid 404 errors.
    """
    global total_requests_used
    try:
        # 1. Get & Clean API Key
        raw_key = api_key or session.get('gemini_api_key') or os.environ.get("GEMINI_API_KEY")
        if not raw_key:
            print("❌ Error: Missing GEMINI_API_KEY")
            return "I am currently offline (API Key missing). Please check my settings."

        clean_key = raw_key.strip().replace('"', '').replace("'", "")

        # 2. Initialize Client
        client = genai.Client(api_key=clean_key)

        # 3. Construct Prompt (Single Block Strategy for stability)
        full_prompt = "You are a helpful, empathetic AI therapist which does not re-enforce the users opinion at the same time not being harsh towards the user , helps user becoming better at solving their problems ensuring they become a better person.\n\n"
        for msg in messages[-6:]: 
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                full_prompt += f"User: {content}\n"
            elif role == 'assistant' or role == 'bot':
                full_prompt += f"Therapist: {content}\n"
        
        full_prompt += "\nTherapist:"

        # 4. Generate Content (Using the NEW syntax)
        # Change this line:
# 4. Generate Content
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=full_prompt,
            config={'max_output_tokens': max_tokens}
        )

        # --- TOKEN MONITORING START ---
# --- UPDATED COUNTDOWN MONITORING ---
        # Every successful response counts as 1 request
        total_requests_used += 1
        requests_left = DAILY_REQUEST_LIMIT - total_requests_used

        if response.usage_metadata:
            tokens_this_msg = getattr(response.usage_metadata, 'total_token_count', 0) or 0
            
            print("\n" + "="*30)
            print(f"📊 API QUOTA REPORT")
            print(f"🔹 Tokens used (this msg): {tokens_this_msg}")
            print(f"✅ Requests remaining today: {requests_left} / {DAILY_REQUEST_LIMIT}")
            print("="*30 + "\n")
        # -----------------------------------

        return response.text

    except Exception as e:
        print(f"❌ Gemini API Error: {str(e)}")
        if "404" in str(e):
            return "Error: Model not found. Please check utils/llm_client.py and ensure 'model' is set to 'gemini-2.5-flash'."
        return "I'm having trouble connecting to my thought process right now."

def validate_groq_api_key(api_key, api_url=None):
    """ Validates the Gemini Key (Renamed for compatibility) """
    try:
        client = genai.Client(api_key=api_key.strip())
        client.models.generate_content(model="gemini-2.5-flash", contents="Test")
        return {"valid": True, "message": "Gemini API key is valid"}
    except Exception as e:
        return {"valid": False, "message": str(e)}