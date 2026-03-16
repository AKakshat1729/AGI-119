#!/usr/bin/env python
"""
Health check and startup verification - Gemini Edition
"""
import os
import sys
from typing import List, Dict, Any
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
        'ASSEMBLYAI_API_KEY': 'AssemblyAI for speech-to-text (Optional)'
    }
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
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
        'pymongo': 'MongoDB driver',
        'google.generativeai': 'Gemini AI Engine',
        'chromadb': 'Vector Memory Database',
        'textblob': 'Sentiment Analysis',
        'nltk': 'Natural Language Toolkit',
        'dotenv': 'Environment variables',
        'pydantic': 'Data validation'
    }
    
    missing = []
    for package, description in packages.items():
        try:
            # Handle packages with dots in names
            base_package = package.split('.')[0]
            __import__(base_package)
            print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: Missing ({description})")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Action Required: Run 'pip install {' '.join(missing)}'")
    
    print()

def check_files():
    """Check if all required files exist"""
    print("🔍 File Check")
    print("-" * 50)
    
    required_files = [
        'app.py',
        'chat_api.py',
        'clinical_resources.py',
        'utils/therapy_llm_client.py',
        'reasoning/long_term_personalized_memory.py',
        '.env'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file}: Missing")
    print()

def validate_app():
    """Try to import and validate the app components"""
    print("🔍 Component Validation")
    print("-" * 50)
    
    try:
        # Check therapy client first
        from utils.therapy_llm_client import TherapyLLMClient, get_llm_response
        print("✅ Therapy LLM Client: Verified")
        
        # Check Memory systems
        from reasoning.long_term_personalized_memory import PersonalizedMemoryModule
        print("✅ Long-Term Memory Module: Verified")
        
        # Check Clinical Intelligence
        from core.clinical_intelligence import MedicalProfileExtractor
        print("✅ Clinical Intelligence: Verified")
        
        return True
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*50)
    print("🏥 AI THERAPIST - STARTUP VERIFICATION")
    print("="*50)
    
    check_environment()
    check_imports()
    check_files()
    
    if validate_app():
        print("\n" + "="*50)
        print("✅ ALL SYSTEMS GO! Your AGI is ready for Lucknow.")
        print("="*50)
        return True
    else:
        print("\n" + "="*50)
        print("❌ CRITICAL ERROR: Check components above.")
        print("="*50)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)