import chromadb
import uuid  # We use this for unique memory IDs
from chromadb.config import Settings

class LongTermMemory:
    def __init__(self, user_id, collection_name="agi_memories"):
        self.user_id = str(user_id)
        # Persistent storage on your laptop
        self.client = chromadb.PersistentClient(path="./agi_memory_vault")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, knowledge, id=None):
        text = str(knowledge)
        if id is None:
            id = str(uuid.uuid4())
        
        # Tag the memory with the user_id so users don't see each other's data
        self.collection.add(
            documents=[text],
            metadatas=[{"user_id": self.user_id}], 
            ids=[id]
        )

    def retrieve(self, query, n_results=5):
        # The 'where' clause is the Gatekeeper that stops the memory leaks
        results = self.collection.query(
            query_texts=[query],
            where={"user_id": self.user_id}, 
            n_results=n_results
        )
        return results

    def get_all(self):
        # Only get memories for this specific user
        return self.collection.get(where={"user_id": self.user_id})