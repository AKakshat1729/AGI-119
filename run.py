#!/usr/bin/env python3
"""
Simple Flask app runner
"""
import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import and run app
try:
    print("\n" + "="*60)
    print("🏥 AI THERAPIST CHATBOT - STARTING")
    print("="*60)
    print("📍 Server: http://127.0.0.1:5000")
    print("🔐 Login: http://127.0.0.1:5000/login")
    print("="*60 + "\n")
    
    from app import app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
