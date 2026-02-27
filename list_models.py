import os
from dotenv import load_dotenv
from google import genai

# 1. Load the key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit()

# 2. Connect
print(f"🔑 Authenticating with key ending in: ...{api_key[-5:]}")
client = genai.Client(api_key=api_key)

print("⏳ Asking Google for available models...")

try:
    # 3. List Models (Simple Loop)
    pager = client.models.list()
    
    print("\n✅ AVAILABLE GEMINI MODELS:")
    print("="*40)
    
    count = 0
    for model in pager:
        # We safely get the name, ignoring Pylance warnings
        name = getattr(model, 'name', 'Unknown')
        
        # Filter: Only show "Gemini" models to keep list clean
        if name and 'gemini' in name.lower():
            # Clean up the name (remove 'models/' prefix if present)
            display_name = name.replace('models/', '')
            print(f"🌟 {display_name}")
            count += 1
            
    if count == 0:
        print("⚠️ No Gemini models found. Your key might have restricted access.")
    else:
        print("="*40)
        print("👉 TIP: Copy one of the names above (like 'gemini-1.5-flash-001')")
        print("        and paste it into utils/llm_client.py")

except Exception as e:
    print(f"❌ Error listing models: {e}")