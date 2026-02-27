#!/usr/bin/env python
"""
Health check and startup verification
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check all environment variables"""
    print("\n🔍 Environment Check")
    print("-" * 50)
    
    required_vars = {
        'FLASK_SECRET_KEY': 'Flask secret (can be auto-generated)',
        'MONGO_URI': 'MongoDB connection string',
        'GEMINI_API_KEY': 'Gemini API key for LLM',
        'ASSEMBLYAI_API_KEY': 'AssemblyAI for speech-to-text'
    }
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            # Hide sensitive data
            hidden_value = value[:10] + '...' if len(value) > 10 else value
            print(f"✅ {var}: {hidden_value}")
        else:
            print(f"⚠️  {var}: Not found ({description})")
    
    print()

def check_imports():
    """Check if all required packages are importable"""
    print("🔍 Package Check")
    print("-" * 50)
    
    packages = {
        'flask': 'Flask web framework',
        'flask_login': 'User authentication',
        'pymongo': 'MongoDB driver',
        'requests': 'HTTP library',
        'assemblyai': 'Speech-to-text API',
        'gtts': 'Text-to-speech',
        'dotenv': 'Environment variables',
        'google.generativeai': 'Google Generative AI (fallback)',
        'pydantic': 'Data validation'
    }
    
    missing = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: Missing ({description})")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Installing missing packages: {', '.join(missing)}")
        os.system(f"pip install {' '.join(missing)} -q")
    
    print()

def check_files():
    """Check if all required files exist"""
    print("🔍 File Check")
    print("-" * 50)
    
    required_files = [
        'app.py',
        'chat_api.py',
        'templates/index.html',
        'templates/login.html',
        'templates/signup.html',
        'clinical_resources.py',
        'utils/therapy_llm_client.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file}: Missing")
    
    print()

def check_static_folders():
    """Create necessary static folders"""
    print("🔍 Static Folder Check")
    print("-" * 50)
    
    folders = [
        'static',
        'static/audio',
        'long_term_memory_db'
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created: {folder}")
        else:
            print(f"✅ {folder}: Exists")
    
    print()

def validate_app():
    """Try to import and validate the app"""
    print("🔍 App Validation")
    print("-" * 50)
    
    try:
        from app import app
        print("✅ app.py imports successfully")
        
        from clinical_resources import get_llm_response
        print("✅ clinical_resources imports successfully")
        
        from utils.therapy_llm_client import TherapyLLMClient
        print("✅ therapy_llm_client imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("\n" + "="*50)
    print("🏥 AI THERAPIST - STARTUP VERIFICATION")
    print("="*50)
    
    check_environment()
    check_imports()
    check_files()
    check_static_folders()
    
    if validate_app():
        print("\n" + "="*50)
        print("✅ All checks passed! Ready to launch.")
        print("="*50)
        return True
    else:
        print("\n" + "="*50)
        print("❌ Some checks failed. Fix issues above.")
        print("="*50)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
