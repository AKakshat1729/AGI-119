import json
from datetime import datetime

def save_memory(text, emotion):
    data = {
        "time": datetime.now().isoformat(),
        "text": text,
        "emotion": emotion
    }

    try:
        with open("memory.json", "r") as f:
            memories = json.load(f)
    except:
        memories = []

    memories.append(data)

    with open("memory.json", "w") as f:
        json.dump(memories, f, indent=2)
