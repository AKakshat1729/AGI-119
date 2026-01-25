from memory.long_term_memory import LongTermMemory
import json
from datetime import datetime

def view_user_history():
    print("📂 PATIENT RECORD VIEWER")
    print("-------------------------")
    
    # 1. Ask which user you want to inspect
    target_user = input("Enter User ID to view (press Enter for 'default'): ").strip()
    if not target_user:
        target_user = "default"

    print(f"\n🔍 Fetching records for User: [{target_user}]...")
    
    try:
        # Connect to that user's memory
        ltm = LongTermMemory(user_id=target_user)
        
        # Pull EVERYTHING
        data = ltm.get_all()
        
        # FIX 1: Safely handle if data is None
        if not data:
            print("❌ No data structure returned.")
            return

        # FIX 2: Safely extract lists using 'or []' to prevent NoneType errors
        documents = data.get('documents') or []
        metadatas = data.get('metadatas') or []
        
        if not documents:
            print("❌ No records found for this user.")
            return

        # 2. Organize the Data
        history = []
        for i in range(len(documents)):
            # Handle potential None in metadata list or index out of bounds
            meta = {}
            if metadatas and i < len(metadatas):
                meta = metadatas[i] or {}
            
            # Try to find a timestamp
            timestamp = meta.get('timestamp') or meta.get('time') or "1970-01-01"
            
            entry = {
                "content": documents[i],
                "meta": meta,
                "time": timestamp
            }
            history.append(entry)

        # 3. SORT BY TIME
        history.sort(key=lambda x: x['time'])

        # 4. Display "Long Term History"
        print(f"\n📜 CHRONOLOGICAL HISTORY ({len(history)} entries)")
        print("--------------------------------------------------")
        for idx, item in enumerate(history):
            content = item['content']
            # Try to parse JSON for cleaner display
            try:
                if isinstance(content, str) and content.strip().startswith('{'):
                    json_content = json.loads(content)
                    if isinstance(json_content, dict) and 'transcript' in json_content:
                        display_text = f"User said: \"{json_content['transcript']}\""
                    else:
                        display_text = str(json_content)
                else:
                    display_text = content
            except:
                display_text = content

            print(f"[{idx+1}] {item['time']} | {display_text}")

        # 5. Display "Working Memory" Context
        print("\n🧠 CURRENT WORKING MEMORY CONTEXT (Last 5 interactions)")
        print("--------------------------------------------------")
        if len(history) > 0:
            recent = history[-5:] 
            for item in recent:
                 # Show a snippet of the content
                 snippet = str(item['content'])[:100].replace('\n', ' ')
                 print(f" -> {snippet}...") 
        else:
            print("(Empty)")

    except Exception as e:
        print(f"\n❌ Error retrieving history: {str(e)}")
        # Debug print to help identify what 'data' actually looks like if it fails
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    view_user_history()