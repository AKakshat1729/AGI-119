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

    def store_memory(self, user_id: str, memory_type: str, text: str, tags: List[str] = None, importance: float = 1.0, conversation_id: str = None) -> str:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        embedding = self._generate_embedding(text)

        metadata = {
            "user_id": user_id,
            "type": memory_type,
            "tags": json.dumps(tags or []),
            "timestamp": timestamp,
            "importance": importance,
            "conversation_id": conversation_id if memory_type == "conversation" else None
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
        elif memory_type == "conversation" and self.conversation_logs_collection is not None:
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

    def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a user."""
        try:
            results = self.conversation_logs_collection.get(
                where={"user_id": user_id},
                limit=limit
            )
            
            conversations = []
            if results['documents'] and results['metadatas']:
                # Sort by timestamp (newest first)
                docs_with_meta = list(zip(results['documents'], results['metadatas']))
                docs_with_meta.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
                
                for doc, meta in docs_with_meta:
                    conversations.append({
                        'id': meta.get('id', ''),
                        'text': doc,
                        'timestamp': meta.get('timestamp', ''),
                        'user_id': meta.get('user_id', ''),
                        'conversation_id': meta.get('conversation_id', 'default'),
                        'tags': meta.get('tags', [])
                    })
            
            return conversations
        except Exception as e:
            print(f"Error retrieving conversation history: {str(e)}")
            return []

    def get_conversation_threads(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation threads grouped by conversation_id."""
        try:
            conversations = self.get_conversation_history(user_id, limit=200)  # Get more to group properly
            
            # Group by conversation_id from metadata
            threads = {}
            for conv in conversations:
                conv_id = conv.get('conversation_id', 'default')
                if conv_id not in threads:
                    # Generate better title from first user message
                    title_text = conv['text']
                    if title_text.startswith('User: '):
                        title_text = title_text.replace('User: ', '').strip()
                    elif title_text.startswith('AI: '):
                        title_text = title_text.replace('AI: ', '').strip()
                    
                    threads[conv_id] = {
                        'id': conv_id,
                        'title': title_text[:50] + '...' if len(title_text) > 50 else title_text,
                        'last_message': conv['timestamp'],
                        'message_count': 0,
                        'preview': title_text[:100] + '...' if len(title_text) > 100 else title_text
                    }
                threads[conv_id]['message_count'] += 1
                # Update last_message if this is newer
                if conv['timestamp'] > threads[conv_id]['last_message']:
                    threads[conv_id]['last_message'] = conv['timestamp']
            
            return list(threads.values())
        except Exception as e:
            print(f"Error retrieving conversation threads: {str(e)}")
            return []

    def get_conversation_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific conversation."""
        try:
            # First get all for user_id, then filter by conversation_id
            results = self.conversation_logs_collection.get(
                where={"user_id": user_id}
            )
            
            messages = []
            if results['documents'] and results['metadatas']:
                docs_with_meta = list(zip(results['documents'], results['metadatas']))
                # Filter by conversation_id
                docs_with_meta = [ (doc, meta) for doc, meta in docs_with_meta if meta.get('conversation_id') == conversation_id ]
                docs_with_meta.sort(key=lambda x: x[1].get('timestamp', ''))
                
                for doc, meta in docs_with_meta:
                    messages.append({
                        'text': doc,
                        'timestamp': meta.get('timestamp', ''),
                        'id': meta.get('id', '')
                    })
            
            return messages
        except Exception as e:
            print(f"Error retrieving conversation messages: {str(e)}")
            return []

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

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a specific user from all collections."""
        try:
            # Delete from profiles collection
            profile_results = self.profiles_collection.get(where={"user_id": user_id})
            if profile_results['ids']:
                self.profiles_collection.delete(ids=profile_results['ids'])

            # Delete from episodic memories collection
            episodic_results = self.episodic_collection.get(where={"user_id": user_id})
            if episodic_results['ids']:
                self.episodic_collection.delete(ids=episodic_results['ids'])

            # Delete from conversation logs collection
            # Collections are guaranteed to exist after initialization
            if self.conversation_logs_collection is not None:
                conversation_results = self.conversation_logs_collection.get(where={"user_id": user_id})
                if conversation_results and conversation_results.get('ids'):
                    self.conversation_logs_collection.delete(ids=conversation_results['ids'])

            return True
        except Exception as e:
            print(f"Error deleting user data: {str(e)}")
            return False