import chromadb
from chromadb.config import Settings
import uuid
import json
from datetime import datetime

class LongTermMemory:
    def __init__(self, user_id="default", persistence_path="long_term_memory_db_Vaibhav_Test"):
        self.user_id = user_id
        # Initialize ChromaDB Client
        self.client = chromadb.PersistentClient(path=persistence_path)
        
        # Create or get the collection for this specific user
        self.collection = self.client.get_or_create_collection(name=f"user_{user_id}_memory")

    def store(self, text, memory_id=None):
        """
        Stores text in the vector database with a timestamp.
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        # Create a timestamp
        current_time = datetime.now().isoformat()

        # Define metadata (This is what was missing before!)
        meta = {
            "timestamp": current_time,
            "type": "conversation_log",
            "user_id": self.user_id
        }

        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[memory_id]
        )
        # print(f"   [Memory] Stored with timestamp: {current_time}")

    def retrieve(self, query, n_results=5):
        """
        Finds memories relevant to the query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def get_all(self):
        """
        Retrieves all memories for history viewing.
        """
        # We fetch a large number to ensure we get everything
        # (Chroma doesn't have a simple 'get_all', so we get the first 1000)
        try:
            return self.collection.get(limit=1000)
        except Exception as e:
            print(f"Error fetching history: {e}")
            return None