import os
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv

# --- Pylance Pacifier ---
# This tells the IDE to treat 'genai' as a dynamic object so it stops flagging imports
genai_client: Any = genai 

def list_available_models():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return

    # Use the alias to satisfy the import symbol check
    genai_client.configure(api_key=api_key)
    
    print("🔍 Listing Gemini Models...")
    try:
        # Using the pacified client here as well
        for model in genai_client.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_available_models()