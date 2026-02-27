#!/usr/bin/env python
"""
Main application launcher
Fixes all known issues and starts the app properly
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required packages
required_packages = [
    'flask', 'flask_login', 'pymongo', 'requests', 'assemblyai', 'gtts'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing packages: {', '.join(missing_packages)}")
    print("Installing missing packages...")
    os.system(f"pip install {' '.join(missing_packages)}")

# Now import the app
try:
    from app import app
    
    print("\n" + "="*60)
    print("🚀 AI THERAPIST CHATBOT - STARTING")
    print("="*60)
    print("Server: http://127.0.0.1:5000")
    print("Login: http://127.0.0.1:5000/login")
    print("Signup: http://127.0.0.1:5000/signup")
    print("="*60 + "\n")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )
    
except Exception as e:
    print(f"❌ Error starting app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
