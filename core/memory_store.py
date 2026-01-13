# core/memory_store.py
# (Bridged Version - Redirects to Main Vector Database)

import json
from datetime import datetime
# Import YOUR powerful memory system
from memory.long_term_memory import LongTermMemory

def save_memory(text, emotion, user_id="default"):
    """
    Saves memory directly to the central Vector Database (ChromaDB)
    instead of a local JSON file.
    """
    print(f"🧠 Unified Memory: Storing '{text}' with emotion '{emotion}'...")
    
    try:
        # Initialize your Long Term Memory
        ltm = LongTermMemory(user_id=user_id)
        
        # Prepare metadata (so we know this came from the teammate's module)
        metadata = {
            "source": "core_module",
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate a unique ID based on time
        mem_id = f"core_{datetime.now().timestamp()}"
        
        # Store in the main Vector DB
        # We store the 'text' combined with 'emotion' for better context
        full_content = f"User felt {emotion}: {text}"
        ltm.store(full_content, mem_id)
        
        print("✅ Successfully saved to LongTermMemory.")
        
    except Exception as e:
        print(f"⚠️ Error saving to Unified Memory: {e}")
        # Fallback: If DB fails, print to console so we don't lose it
        print(f"FALLBACK MEMORY: {text} [{emotion}]")