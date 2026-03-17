import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Any

# --- Pylance Pacifier ---
# This stops VS Code from complaining about "configure", "list_models", etc.
genai_client: Any = genai 

def check_gemini_setup():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        return

    # Use genai_client to satisfy Pylance (Line 17)
    genai_client.configure(api_key=api_key)
    
    print("✅ API Key configured.")
    print("🔍 Checking available models...")

    try:
        # Use genai_client to satisfy Pylance (Line 22)
        models = genai_client.list_models()
        available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        if available_models:
            print(f"✅ Connection successful! Found {len(available_models)} generative models.")
            print(f"📋 Primary model: {available_models[0]}")
            
            # Use genai_client to satisfy Pylance (Line 39)
            test_model = genai_client.GenerativeModel('gemini-3-flash-preview')
            print("🧪 Testing a simple generation...")
            response = test_model.generate_content("Say 'System Ready'")
            print(f"🤖 AI Response: {response.text.strip()}")
        else:
            print("❌ No generative models found. Check your API permissions.")
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")

if __name__ == "__main__":
    check_gemini_setup()