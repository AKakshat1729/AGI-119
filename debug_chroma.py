import chromadb
client = chromadb.PersistentClient(path="long_term_memory_db")
cols = client.list_collections()
for c in cols:
    print(f"Collection: {c.name}, Count: {c.count()}")
