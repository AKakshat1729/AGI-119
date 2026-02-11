import os
from dotenv import load_dotenv
from google import genai # <--- The NEW library

# 1. Load environment variables
load_dotenv()

# 2. Get the key
key = os.getenv("GEMINI_API_KEY")

print(f"\n🔍 DEBUG INFO:")
print(f"✅ Key Found: {key[:5]}...{key[-5:]}" if key else "❌ Key Missing")

if key:
    try:
        # 3. Initialize the New Client
        client = genai.Client(api_key=key)
        
        print("⏳ Contacting Google API (using google-genai)...")
        
        # 4. Generate Content (The new syntax)
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents="Say 'Hello' if this works."
        )
        
        print(f"✅ Response: {response.text}")
        print("🎉 SUCCESS! The key and library are working.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   -> If this is a 404, check if 'gemini-1.5-flash' is available in your region.")
else:
    print("❌ ERROR: GEMINI_API_KEY not found in .env")