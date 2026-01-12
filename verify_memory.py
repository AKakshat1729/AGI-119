from memory.long_term_memory import LongTermMemory
from core.memory_store import save_memory 
import time

def run_memory_diagnostics():
    print("🧠 STARTING MEMORY DIAGNOSTICS...\n")
    
    # --- TEST 1: The Bridge Test ---
    print("test 1: Testing Unified Memory Bridge...")
    test_phrase = f"Test Memory {int(time.time())}: The bridge is working!"
    
    # 1. Save using the TEAMMATE'S function
    print(f"   -> Saving via core.memory_store: '{test_phrase}'")
    save_memory(test_phrase, "excited", user_id="default")
    
    # 2. Retrieve using YOUR function
    print("   -> Attempting to read back via LongTermMemory...")
    ltm = LongTermMemory(user_id="default")
    
    # Search specifically for the test phrase
    results = ltm.retrieve(test_phrase, n_results=1)
    
    # FIX: Use 'or []' to ensure we never loop through None
    raw_docs = results.get('documents') or []
    
    found = False
    # Flatten the list of lists safely
    if raw_docs:
        docs = [d for sublist in raw_docs for d in sublist]
        for doc in docs:
            if test_phrase in doc:
                found = True
                print(f"   ✅ SUCCESS! Found entry: '{doc}'")
                break
    
    if not found:
        print("   ❌ FAILURE: Could not retrieve the memory saved by the bridge.")

    print("\n------------------------------------------------\n")

    # --- TEST 2: Recalling Old Files ---
    print("test 2: Reading Recent History (Recall Check)...")
    history = ltm.retrieve("I", n_results=10)
    
    # FIX: Use 'or []' here too
    history_docs = history.get('documents') or []
    
    if history_docs:
        docs = [d for sublist in history_docs for d in sublist]
        print(f"   -> Found {len(docs)} memories in the database:")
        for i, doc in enumerate(docs):
            print(f"      [{i+1}] {doc}")
    else:
        print("   ⚠️ Database appears empty or returned no results.")

if __name__ == "__main__":
    run_memory_diagnostics()