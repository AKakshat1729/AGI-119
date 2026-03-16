import chromadb
from chromadb.utils import embedding_functions
from typing import Any, List, Optional

print("🔍 Verifying ChromaDB vector storage...")

# We use Any to bypass the strict Contravariant protocol check from Pylance
emb_fn: Any = None

try:
    # Initialize the embedding function safely
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    print("✅ SentenceTransformer embedding loaded OK")
except Exception as e:
    print(f"⚠️  SentenceTransformer failed: {e}. Using default embedding.")
    emb_fn = None

# 1. Check personalized memory DB
print("\n--- personalized_vector_db ---")
try:
    client1 = chromadb.PersistentClient(path="personalized_vector_db")
    collections1 = client1.list_collections()
    print(f"Collections: {[c.name for c in collections1]}")
    
    if collections1:
        # Pylance Pacifier: Explicitly casting emb_fn to Any in the call
        col = client1.get_collection(name="user_memories", embedding_function=emb_fn)
        count = col.count()
        print(f"user_memories count: {count}")
        
        if count > 0:
            # Safety Wrap: Ensure items is not None before subscripting
            items = col.get(limit=3)
            if items:
                # Use .get() and fallback to empty list to satisfy Pylance
                docs: List[Any] = items.get('documents') or []
                print(f"Sample docs: {docs[:3]}")
        else:
            print("⚠️  Collection is EMPTY — no memories extracted yet. Try chatting first.")
    else:
        print("ℹ️  No collections yet.")
except Exception as e:
    print(f"❌ Error in Personalized DB: {e}")

# 2. Check working memory DB
print("\n--- working_memory_db ---")
try:
    client2 = chromadb.PersistentClient(path="working_memory_db")
    collections2 = client2.list_collections()
    print(f"Collections: {[c.name for c in collections2]}")
    for c in collections2:
        col2 = client2.get_collection(name=str(c.name))
        print(f"  {c.name}: {col2.count()} items")
except Exception as e:
    print(f"❌ Error in Working DB: {e}")

# 3. Check server long-term memory
print("\n--- long_term_memory_db ---")
try:
    client3 = chromadb.PersistentClient(path="long_term_memory_db")
    collections3 = client3.list_collections()
    print(f"Collections: {[c.name for c in collections3]}")
    for c in collections3:
        col3 = client3.get_collection(name=str(c.name))
        print(f"  {c.name}: {col3.count()} items")
except Exception as e:
    print(f"❌ Error in Long-term DB: {e}")

print("\n✅ Verification complete.")