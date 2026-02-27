import chromadb
from chromadb.utils import embedding_functions

print("🔍 Verifying ChromaDB vector storage...")

try:
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    print("✅ SentenceTransformer embedding loaded OK")
except Exception as e:
    print(f"⚠️  SentenceTransformer failed: {e}. Using default embedding.")
    emb_fn = None

# 1. Check personalized memory DB (uses SentenceTransformer)
print("\n--- personalized_vector_db ---")
try:
    client1 = chromadb.PersistentClient(path="personalized_vector_db")
    collections1 = client1.list_collections()
    print(f"Collections: {[c.name for c in collections1]}")
    if collections1:
        col = client1.get_collection(name="user_memories", embedding_function=emb_fn)
        count = col.count()
        print(f"user_memories count: {count}")
        if count > 0:
            items = col.get(limit=3)
            print(f"Sample docs: {items['documents'][:3]}")
        else:
            print("⚠️  Collection is EMPTY — no memories extracted yet. Try chatting first.")
    else:
        print("ℹ️  No collections yet.")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Check working memory DB (NO custom embedding — uses default)
print("\n--- working_memory_db ---")
try:
    client2 = chromadb.PersistentClient(path="working_memory_db")
    collections2 = client2.list_collections()
    print(f"Collections: {[c.name for c in collections2]}")
    for c in collections2:
        col2 = client2.get_collection(name=c.name)  # NO emb_fn — avoids conflict
        print(f"  {c.name}: {col2.count()} items")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Check server long-term memory (NO custom embedding)
print("\n--- long_term_memory_db ---")
try:
    client3 = chromadb.PersistentClient(path="long_term_memory_db")
    collections3 = client3.list_collections()
    print(f"Collections: {[c.name for c in collections3]}")
    for c in collections3:
        col3 = client3.get_collection(name=c.name)
        print(f"  {c.name}: {col3.count()} items")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n✅ Verification complete.")
