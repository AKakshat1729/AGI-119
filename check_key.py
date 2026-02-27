import os
from dotenv import load_dotenv
import google.generativeai as genai  # Using the installed google-generativeai package

# 1. Load environment variables
load_dotenv()

# 2. Get the key
key = os.getenv("GEMINI_API_KEY")

print(f"\n🔍 DEBUG INFO:")
print(f"✅ Key Found: {key[:5]}...{key[-5:]}" if key else "❌ Key Missing")

if key:
    try:
        # 3. Configure Gemini
        genai.configure(api_key=key)

        print("⏳ Contacting Google API (using google-generativeai)...")

        # 4. List available models
        models = genai.list_models()
        available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        print(f"📋 Available models: {available[:5]}")

        # 5. Pick the best available model
        preferred = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        chosen = None
        for p in preferred:
            if any(p in m for m in available):
                chosen = next(m for m in available if p in m)
                break

        if not chosen and available:
            chosen = available[0]

        if chosen:
            print(f"🎯 Testing model: {chosen}")
            model = genai.GenerativeModel(chosen)
            response = model.generate_content("Say 'Hello' if this works.")
            print(f"✅ Response: {response.text}")
            print("🎉 SUCCESS! The key and library are working.")
        else:
            print("❌ No usable models found for your API key.")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ ERROR: GEMINI_API_KEY not found in .env")