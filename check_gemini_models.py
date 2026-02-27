"""
check_gemini_models.py
─────────────────────
Utility to probe all available Gemini models and automatically switch
to a working one when the current model hits a quota / rate-limit error.

Usage (standalone):
    python check_gemini_models.py

Usage (as module):
    from check_gemini_models import find_working_model, update_env_model
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Priority-ordered candidate models — fastest / cheapest first
CANDIDATE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-pro",
]

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def _configure(api_key: str = None):
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if key:
        genai.configure(api_key=key.strip())
    return key


def update_env_model(model_name: str, env_path: str = ENV_PATH) -> bool:
    """Write LLM_MODEL=<model_name> into .env and update os.environ in-process."""
    try:
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("LLM_MODEL="):
                lines[i] = f"LLM_MODEL={model_name}\n"
                found = True
                break
        if not found:
            lines.append(f"LLM_MODEL={model_name}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.environ["LLM_MODEL"] = model_name
        print(f"[MODEL SWITCH] ✓ .env updated: LLM_MODEL={model_name}")
        return True
    except Exception as e:
        print(f"[MODEL SWITCH] Failed to update .env: {e}")
        return False


def find_working_model(api_key: str = None, candidates: list = None) -> str | None:
    """
    Test each candidate model with a minimal prompt.
    Returns the first model name that succeeds, or None if all fail.
    Also updates .env automatically when a working model is found.
    """
    key = _configure(api_key)
    if not key:
        print("[MODEL SWITCH] No API key available — cannot probe models.")
        return None

    probe_list = candidates or CANDIDATE_MODELS
    print(f"[MODEL SWITCH] Probing {len(probe_list)} models for a working one...")

    for model_name in probe_list:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                "Reply with the single word: OK",
                generation_config={"max_output_tokens": 5, "temperature": 0.0},
            )
            if response and response.text:
                print(f"[MODEL SWITCH] ✓ '{model_name}' is working!")
                update_env_model(model_name)
                return model_name
        except Exception as e:
            err = str(e)
            if "quota" in err.lower() or "429" in err or "rate" in err.lower():
                print(f"[MODEL SWITCH] ✗ '{model_name}' quota/rate-limited — trying next.")
            elif "not found" in err.lower() or "404" in err:
                print(f"[MODEL SWITCH] ✗ '{model_name}' not available — trying next.")
            else:
                print(f"[MODEL SWITCH] ✗ '{model_name}' error: {err[:80]}")

    print("[MODEL SWITCH] ✗ No working model found. Check your API key / quota.")
    return None


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Checking available Gemini models...\n")
    try:
        _configure()
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", [])
            if "generateContent" in methods:
                print(f"  • {model.name}  [{model.display_name}]")
    except Exception as e:
        print(f"[ERROR] Could not list models: {e}")

    print("\n[INFO] Finding a working model...\n")
    result = find_working_model()
    if result:
        print(f"\n✅ Active model set to: {result}")
    else:
        print("\n❌ Could not find a working model.")
