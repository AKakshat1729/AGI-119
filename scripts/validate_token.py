import os
import sys
from dotenv import load_dotenv

# 1. Fix the pathing so the script can see the 'utils' folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # We use the Gemini-based client we cleaned up earlier
    from utils.llm_client import validate_gemini_api_key
    _llm_ready = True
except ImportError as e:
    _llm_ready = False
    print(f"[ERROR] Could not find llm_client: {e}")

def run_validation():
    print("--- AGI Token Validation ---")
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("[FAIL] No GEMINI_API_KEY found in .env file.")
        return

    if _llm_ready:
        print(f"Validating key: {api_key[:4]}...{api_key[-4:]}")
        result = validate_gemini_api_key(api_key)
        
        if result.get("valid"):
            print("[SUCCESS] Gemini Token is active and responding!")
        else:
            print(f"[FAIL] API Error: {result.get('message')}")
            print(f"Details: {result.get('error')}")
    else:
        print("[SKIP] Validation logic missing. Ensure utils/llm_client.py exists.")

if __name__ == "__main__":
    run_validation()