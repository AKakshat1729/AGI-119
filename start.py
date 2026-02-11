#!/usr/bin/env python3
"""
Startup script for AGI-119 Application
Runs both FastAPI Chat Server and Flask Web Server
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    """Start both FastAPI and Flask servers"""
    
    print("=" * 60)
    print("🤖  AGI-119 Application Startup")
    print("=" * 60)
    
    # Check if required environment variables are set
    print("\n📋 Checking environment configuration...")
    
    if not (os.getenv("GROQ_API_KEY")):
        print("⚠️  WARNING: No Groq API key set in .env file")
        print("   You can update your Groq API key in Settings after signup")
    
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("⚠️  WARNING: ASSEMBLYAI_API_KEY not set in .env file")
    
    # Create processes list
    processes = []
    
    try:
        # Start FastAPI Chat Server
        print("\n🚀 Starting FastAPI Chat Server...")
        print("   Listening on: http://localhost:8000")
        chat_process = subprocess.Popen(
            [sys.executable, "chat_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("FastAPI Chat Server", chat_process))
        time.sleep(2)  # Give FastAPI time to start
        
        # Start Flask Web Server  
        print("\n🌐 Starting Flask Web Server...")
        print("   Visit: http://localhost:5000")
        print("\n   Login with your credentials or create a new account")
        flask_process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Flask Web Server", flask_process))
        
        print("\n" + "=" * 60)
        print("✅ Application Started Successfully!")
        print("=" * 60)
        print("\n📝 Configuration Notes:")
        print("   • Set GROQ_API_KEY in .env or update in Settings")
        print("   • FastAPI runs on port 8000 (API server)")
        print("   • Flask runs on port 5000 (Web UI)")
        print("\n💡 Tips:")
        print("   • Press Ctrl+C to stop the application")
        print("   • Check the console for any errors")
        print("   • Settings page allows API key updates")
        print("\n" + "=" * 60 + "\n")
        
        # Keep the processes running
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n❌ {name} has stopped!")
                    print("Shutting down application...")
                    sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down application...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ Application stopped")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        for name, process in processes:
            if process.poll() is None:
                process.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()
