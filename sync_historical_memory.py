import os
import json
from typing import Any, List, Dict, Set
from dotenv import load_dotenv
from api.memory_store import ServerMemoryStore
from reasoning.long_term_personalized_memory import PersonalizedMemoryModule
from utils.llm_client import generate_chat_response

# Load environment
load_dotenv()

def sync_all_users():
    print("--- Starting Bulk Historical Memory Sync ---")
    
    # 1. Initialize stores
    # Using 'Any' to prevent Pylance from flagging internal ChromaDB types
    memory_store: Any = ServerMemoryStore()
    pers_memory: Any = PersonalizedMemoryModule()
    
    # 2. Identify all collections that contain conversation logs
    # Pylance Pacifier: Ensuring we treat the client as a dynamic object
    client_api: Any = memory_store.client
    all_col_names = [c.name for c in client_api.list_collections()]
    log_cols = [name for name in all_col_names if "conversation_logs" in name]
    
    print(f"Scanning collections: {log_cols}")
    
    processed_conv_ids: Set[str] = set()

    for col_name in log_cols:
        print(f"\n--- Scanning Collection: {col_name} ---")
        col = client_api.get_collection(name=col_name)
        results = col.get(limit=10000)
        
        # --- Safety Guard for Results ---
        # ChromaDB results can be None, so we force them to empty lists if missing
        ids = results.get('ids') or []
        metas = results.get('metadatas') or []
        documents = results.get('documents') or []
        
        if not ids:
            continue

        # Extract unique users and force them to strings to satisfy Pylance
        found_ids = [str(m.get('user_id') or "") for m in metas]
        user_ids = list(set([uid for uid in found_ids if uid]))
        print(f"  Found users in {col_name}: {user_ids}")
        
        for user_id in user_ids:
            # Ensure user_id is strictly a string
            safe_user_id = str(user_id)
            print(f"  Processing User: {safe_user_id}")

            user_data = []
            # Using the safety-wrapped 'documents' list
            for i, doc in enumerate(documents):
                # Ensure meta is not None before subscripting
                meta = metas[i] if i < len(metas) else {}
                
                if meta.get('user_id') == safe_user_id:
                    user_data.append({
                        "id": ids[i] if i < len(ids) else "unknown",
                        "text": str(doc or ""),
                        "meta": meta
                    })
            
            # Group by conversation
            convo_groups: Dict[str, List[Dict]] = {}
            for item in user_data:
                cid = str(item['meta'].get('conversation_id', 'unknown'))
                if cid not in convo_groups:
                    convo_groups[cid] = []
                convo_groups[cid].append(item)
            
            print(f"    Found {len(convo_groups)} threads.")

            for cid, items in convo_groups.items():
                if cid in processed_conv_ids:
                    continue
                
                # Sort items by timestamp safely
                items.sort(key=lambda x: str(x['meta'].get('timestamp', '')))
                
                transcript = ""
                for item in items:
                    text = str(item['text'])
                    role = "User" if text.startswith("User:") else "Assistant"
                    clean_text = text.replace("User: ", "").replace("AI: ", "").replace("Assistant: ", "")
                    transcript += f"{role}: {clean_text}\n"
                
                if len(transcript) > 50:
                    print(f"    Analyzing session {cid} ({len(items)} messages)...")
                    api_key = os.environ.get("GEMINI_API_KEY")
                    
                    # Force user_id to string and wrap analysis in try-block
                    try:
                        pers_memory.analyze_historical_data(
                            safe_user_id, 
                            [{"conversation_id": cid, "transcript": transcript}], 
                            generate_chat_response, 
                            api_key=api_key
                        )
                        processed_conv_ids.add(cid)
                    except Exception as e:
                        print(f"    ❌ Error analyzing {cid}: {e}")

    print("\n--- Bulk Historical Memory Sync Completed ---")

if __name__ == "__main__":
    sync_all_users()