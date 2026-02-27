import os
import json
from dotenv import load_dotenv
from api.memory_store import ServerMemoryStore
from reasoning.long_term_personalized_memory import PersonalizedMemoryModule
from utils.llm_client import generate_chat_response

# Load environment
load_dotenv()

def sync_all_users():
    print("--- Starting Bulk Historical Memory Sync ---")
    
    # 1. Initialize stores
    memory_store = ServerMemoryStore()
    pers_memory = PersonalizedMemoryModule()
    
    # 2. Identify all collections that contain conversation logs
    all_col_names = [c.name for c in memory_store.client.list_collections()]
    log_cols = [name for name in all_col_names if "conversation_logs" in name]
    
    print(f"Scanning collections: {log_cols}")
    
    # Track which sessions we've processed to avoid duplicate analysis
    processed_conv_ids = set()

    for col_name in log_cols:
        print(f"\n--- Scanning Collection: {col_name} ---")
        col = memory_store.client.get_collection(name=col_name)
        results = col.get(limit=10000)
        
        if not results or not results['ids']:
            continue

        # Extract unique users in this collection
        found_ids = [m.get('user_id') for m in results['metadatas']]
        user_ids = list(set([uid for uid in found_ids if uid]))
        print(f"  Found users in {col_name}: {user_ids}")
        
        for user_id in user_ids:
            print(f"  Processing User: {user_id}")

            
            # Reconstruct threads from THIS collection
            # Group by conversation_id manually since we're bypassing the store's filter
            user_data = []
            for i, doc in enumerate(results['documents']):
                meta = results['metadatas'][i]
                if meta.get('user_id') == user_id:
                    user_data.append({
                        "id": results['ids'][i],
                        "text": doc,
                        "meta": meta
                    })
            
            # Group by conversation
            convo_groups = {}
            for item in user_data:
                cid = item['meta'].get('conversation_id', 'unknown')
                if cid not in convo_groups:
                    convo_groups[cid] = []
                convo_groups[cid].append(item)
            
            print(f"    Found {len(convo_groups)} threads.")

            for cid, items in convo_groups.items():
                if cid in processed_conv_ids:
                    continue
                
                # Sort items by timestamp if available
                items.sort(key=lambda x: x['meta'].get('timestamp', ''))
                
                transcript = ""
                for item in items:
                    text = item['text']
                    role = "User" if text.startswith("User:") else "Assistant"
                    clean_text = text.replace("User: ", "").replace("AI: ", "").replace("Assistant: ", "")
                    transcript += f"{role}: {clean_text}\n"
                
                if len(transcript) > 50:
                    print(f"    Analyzing session {cid} ({len(items)} messages)...")
                    # Analysis
                    api_key = os.environ.get("GEMINI_API_KEY")
                    pers_memory.analyze_historical_data(user_id, [{"conversation_id": cid, "transcript": transcript}], generate_chat_response, api_key=api_key)
                    processed_conv_ids.add(cid)


    print("\n--- Bulk Historical Memory Sync Completed ---")

if __name__ == "__main__":
    sync_all_users()
