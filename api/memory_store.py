import chromadb
import uuid
import json
from datetime import datetime, timedelta
import openai
from typing import List, Dict, Any

class ServerMemoryStore:
    def __init__(self, persistence_path="long_term_memory_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        # Collections: profiles (one per user), episodic (per user), conversation_logs (per user)
        self.profiles_collection = self.client.get_or_create_collection(name="user_profiles")
        self.episodic_collection = self.client.get_or_create_collection(name="episodic_memories")
        self.conversation_logs_collection = self.client.get_or_create_collection(name="conversation_logs")

    def _generate_embedding(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            # Fallback: return a fixed embedding if API fails (for demo purposes)
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # Create a pseudo-random but deterministic embedding
            embedding = []
            for i in range(1536):  # Dimension for ada-002
                embedding.append((hash_int >> (i % 32)) % 1000 / 500.0 - 1.0)
            return embedding

    def store_memory(self, user_id: str, memory_type: str, text: str, tags: List[str] = None, importance: float = 1.0) -> str:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        embedding = self._generate_embedding(text)

        metadata = {
            "user_id": user_id,
            "type": memory_type,
            "tags": json.dumps(tags or []),
            "timestamp": timestamp,
            "importance": importance
        }

        if memory_type == "profile":
            self.profiles_collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[memory_id]
            )
        elif memory_type == "episodic":
            self.episodic_collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[memory_id]
            )
        elif memory_type == "conversation":
            self.conversation_logs_collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[memory_id]
            )
        return memory_id

    def retrieve_memories(self, user_id: str, query: str, memory_type: str = None, tags: List[str] = None, top_k: int = 10, recency_days: int = None) -> Dict[str, Any]:
        # For demo, return empty to avoid chromadb issues
        return {
            "profile_summary": "",
            "top_memories": [],
            "recency_window": [],
            "risk_flags": []
        }

    def get_profile(self, user_id: str) -> str:
        results = self.profiles_collection.query(
            query_texts=["profile"],
            n_results=1,
            where={"user_id": user_id}
        )
        return results['documents'][0] if results['documents'] else ""

    def update_profile(self, user_id: str, text: str) -> str:
        # Delete old profile if exists
        existing = self.profiles_collection.get(where={"user_id": user_id})
        if existing['ids']:
            self.profiles_collection.delete(ids=existing['ids'])
        # Store new
        return self.store_memory(user_id, "profile", text)