import chromadb
import uuid
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

class ServerMemoryStore:
    def __init__(self, persistence_path="long_term_memory_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        
        # Provider Configuration
        self.providers = {
            "gemini": {
                "model": "models/text-embedding-004",
                "dimension": 768,
                "collection_suffix": "_768"
            },
            "openai": {
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "collection_suffix": "_1536"
            },
            "hash": {
                "model": "md5-hash",
                "dimension": 768,
                "collection_suffix": "_hash_768"
            }
        }
        
        # Initialize Collections for ALL providers
        self.collections = {}
        for provider, config in self.providers.items():
            suffix = config["collection_suffix"]
            self.collections[provider] = {
                "profiles": self.client.get_or_create_collection(name=f"user_profiles{suffix}"),
                "episodic": self.client.get_or_create_collection(name=f"episodic_memories{suffix}"),
                "logs": self.client.get_or_create_collection(name=f"conversation_logs{suffix}"),
                "clinical": self.client.get_or_create_collection(name=f"clinical_knowledge{suffix}")
            }

        # Set Initial Active Provider
        self.active_provider = "gemini"
        
        # Auto-detect the provider that actually has data (handles restart after quota fallback)
        self._auto_detect_provider()
        
        # Check environment for OpenAI availability
        try:
            import openai
            self.openai_available = True
        except ImportError:
            self.openai_available = False

    def _auto_detect_provider(self):
        """At startup, switch to the provider that has the most logged data."""
        try:
            best_provider = "gemini"
            best_count = 0
            for provider, cols in self.collections.items():
                count = cols["logs"].count()
                if count > best_count:
                    best_count = count
                    best_provider = provider
            if best_provider != self.active_provider:
                print(f"[MEMORY] Auto-detected active provider: {best_provider} ({best_count} log entries)")
                self.active_provider = best_provider
        except Exception as e:
            print(f"[MEMORY] Provider auto-detect failed: {e}")

    def _switch_provider(self, new_provider):
        print(f"[MEMORY] Switching provider from {self.active_provider} to {new_provider}")
        self.active_provider = new_provider

    def _generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generates embedding using active provider. 
        Handles PermissionDenied/Suspension by automatic fallback.
        Returns: { 'vector': List[float], 'metadata': Dict }
        """
        
        # 1. Try GEMINI
        if self.active_provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("Missing GEMINI_API_KEY")
                
                genai.configure(api_key=api_key)
                result = genai.embed_content(
                    model=self.providers["gemini"]["model"],
                    content=text,
                    task_type="retrieval_document",
                    title="Embedding of text"
                )
                
                if 'embedding' in result:
                    return {
                        "vector": result['embedding'],
                        "metadata": {
                            "provider": "gemini",
                            "model": self.providers["gemini"]["model"],
                            "dimension": self.providers["gemini"]["dimension"]
                        }
                    }
                    
            except Exception as e:
                error_str = str(e).lower()
                # Detect quota exhaustion, rate limits, permission errors
                if any(kw in error_str for kw in ["permission denied", "suspended", "403", "429", "quota", "rate limit", "resource_exhausted", "free_tier"]):
                    print(f"[MEMORY] Gemini quota/permission error, switching to hash: {e}")
                    self._switch_provider("hash")
                    return self._generate_embedding(text)
                else:
                    print(f"[MEMORY WARNING] Gemini Error: {e}. Falling back to hash.")
                    self._switch_provider("hash")
                    return self._generate_embedding(text)

        # 2. Try OPENAI
        if self.active_provider == "openai":
            try:
                import openai
                # Ensure key is set
                if not os.environ.get("OPENAI_API_KEY"):
                    print("[MEMORY] No OpenAI Key found, falling back to Hash")
                    self._switch_provider("hash")
                    return self._generate_embedding(text)

                response = openai.Embedding.create(
                    input=text, 
                    model=self.providers["openai"]["model"]
                )
                return {
                    "vector": response['data'][0]['embedding'],
                    "metadata": {
                        "provider": "openai",
                        "model": self.providers["openai"]["model"],
                        "dimension": self.providers["openai"]["dimension"]
                    }
                }
            except Exception as e:
                print(f"[MEMORY ERROR] OpenAI Error: {e}")
                self._switch_provider("hash")
                return self._generate_embedding(text)

        # 3. Fallback HASH
        # Default/Last Resort
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        embedding = []
        dim = self.providers["hash"]["dimension"]
        for i in range(dim):  
            embedding.append((hash_int >> (i % 32)) % 1000 / 500.0 - 1.0)
            
        return {
            "vector": embedding,
            "metadata": {
                "provider": "hash",
                "model": "md5",
                "dimension": dim
            }
        }

    def store_memory(self, user_id: str, memory_type: str, text: str, tags: List[str] = None, importance: float = 1.0, conversation_id: str = None) -> str:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Get Embedding & Provider Metadata
        embed_result = self._generate_embedding(text)
        embedding = embed_result["vector"]
        provider_meta = embed_result["metadata"]
        
        # Base Metadata
        metadata = {
            "user_id": user_id,
            "type": memory_type,
            "tags": json.dumps(tags or []),
            "timestamp": timestamp,
            "importance": importance,
            "conversation_id": conversation_id, # Allow for all types
            # Add Provider Metadata for tracking
            "embed_provider": provider_meta["provider"],
            "embed_model": provider_meta["model"],
            "embed_dim": provider_meta["dimension"]
        }

        # Select Collection based on Active Provider
        active_cols = self.collections[self.active_provider]
        
        if memory_type in ["profile", "core_insight"]:
            active_cols["profiles"].add(documents=[text], metadatas=[metadata], embeddings=[embedding], ids=[memory_id])
        elif memory_type == "episodic":
            active_cols["episodic"].add(documents=[text], metadatas=[metadata], embeddings=[embedding], ids=[memory_id])
        elif memory_type == "conversation":
            active_cols["logs"].add(documents=[text], metadatas=[metadata], embeddings=[embedding], ids=[memory_id])
        else:
            # Fallback for any other type
            active_cols["episodic"].add(documents=[text], metadatas=[metadata], embeddings=[embedding], ids=[memory_id])
            
        return memory_id

    def retrieve_memories(self, user_id: str, query: str = "", memory_type: str = "episodic", tags: List[str] = None, top_k: int = 5, recency_days: int = None, filter_tags: List[str] = None) -> List[Dict[str, Any]]:
        try:
            # Construct explicit $and filter for ChromaDB
            where_conditions = [{"user_id": user_id}]
            if memory_type:
                where_conditions.append({"type": memory_type})
            
            # Use filter_tags if provided (standardize the arg name)
            effective_tags = filter_tags or tags
            
            if len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}
            else:
                where_filter = where_conditions[0]
            
            # Select Collection based on Active Provider
            active_cols = self.collections[self.active_provider]
            
            collection = active_cols["episodic"]
            if memory_type == "profile":
                collection = active_cols["profiles"]
            elif memory_type == "conversation":
                collection = active_cols["logs"]

            # If query is empty, just get the most recent ones without semantic search
            if not query or query.strip() == "":
                results = collection.get(
                    where=where_filter,
                    limit=top_k
                )
                # Format results to match query output structure
                # .get() returns a dict with 'documents', 'metadatas', 'ids'
            else:
                # Generate Embedding
                embed_result = self._generate_embedding(query)
                embedding = embed_result["vector"]
                
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    where=where_filter
                )
            
            memories = []
            if results and results.get('documents'):
                # results['documents'] is List[str] for .get() but List[List[str]] for .query()
                docs = results['documents']
                metas = results['metadatas']
                ids = results['ids']
                
                # Normalize structure if it's from .query()
                if isinstance(docs[0], list) and not isinstance(docs, str):
                    docs = docs[0]
                    metas = metas[0]
                    ids = ids[0]
                    distances = results.get('distances', [[]])[0]
                else:
                    distances = [0] * len(docs)

                for i, doc in enumerate(docs):
                    if i >= len(metas): break
                    
                    meta = metas[i]
                    mem_id = ids[i]
                    dist = distances[i]
                    
                    # Manual tag filtering if requested (since Chroma doesn't support $contains for JSON-safe strings well)
                    if effective_tags:
                        stored_tags = json.loads(meta.get('tags', '[]'))
                        if not any(tag in stored_tags for tag in effective_tags):
                            continue

                    if recency_days and 'timestamp' in meta:
                        try:
                            timestamp = datetime.fromisoformat(meta['timestamp'])
                            if datetime.now() - timestamp > timedelta(days=recency_days):
                                continue
                        except: pass
                            
                    memories.append({
                        "id": mem_id,
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "type": meta.get('type', memory_type)
                    })
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_clinical_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        try:
            embed_result = self._generate_embedding(query)
            embedding = embed_result["vector"]
            
            active_cols = self.collections[self.active_provider]
            collection = active_cols["clinical"]
            
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            if results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"Error retrieving clinical: {e}")
            return []

    def init_clinical_knowledge(self, clinical_data: Dict[str, Any]):
        """Initialize clinical knowledge base using data from clinical_resources.py"""
        try:
            active_cols = self.collections[self.active_provider]
            col = active_cols["clinical"]
            
            if col.count() > 0:
                print(f"Clinical Knowledge Base ({self.active_provider}) already initialized.")
                return

            print(f"Initializing Clinical Knowledge Base for {self.active_provider}...")
            
            ids = []
            docs = []
            metas = []
            
            def process_item(key, value, category="general"):
                if isinstance(value, str):
                    docs.append(f"{category} - {key}: {value}")
                    metas.append({"category": category, "topic": key, "source": "Clinical Protocols"})
                    ids.append(str(uuid.uuid4()))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            docs.append(f"{category} - {key}: {item}")
                            metas.append({"category": category, "topic": key, "source": "Evidence Based Practice"})
                            ids.append(str(uuid.uuid4()))
                elif isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                         process_item(sub_k, sub_v, category=key)

            for main_k, main_v in clinical_data.items():
                process_item(main_k, main_v, category=main_k)

            if docs:
                batch_size = 50
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i+batch_size]
                    batch_metas = metas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    
                    # Embed batch
                    # Note: _generate_embedding is single text, need to loop or batch if supported
                    # Chroma usually handles batch if passed embedding function, but we are manual
                    batch_embeddings = []
                    for doc in batch_docs:
                         res = self._generate_embedding(doc)
                         batch_embeddings.append(res["vector"])
                    
                    col.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas,
                        embeddings=batch_embeddings
                    )
            print(f"Clinical Knowledge Base populated with {len(docs)} items.")
            
        except Exception as e:
            print(f"Error initializing clinical knowledge: {e}")

    # --- Passthrough/Helper methods ---
    def get_profile(self, user_id: str) -> str:
        # Try retrieving from active provider
        active_cols = self.collections[self.active_provider]
        results = active_cols["profiles"].query(
            query_texts=["profile"], n_results=1, where={"user_id": user_id}
        )
        return results['documents'][0] if results['documents'] else ""

    def update_profile(self, user_id: str, text: str) -> str:
        # Delete old profile from ACTIVE provider only 
        active_cols = self.collections[self.active_provider]
        existing = active_cols["profiles"].get(where={"user_id": user_id})
        if existing['ids']:
            active_cols["profiles"].delete(ids=existing['ids'])
        return self.store_memory(user_id, "profile", text)

    def delete_user_data(self, user_id: str) -> bool:
        try:
            # Delete from ALL providers to be thorough
            for provider in self.collections:
                cols = self.collections[provider]
                for key, col in cols.items():
                    res = col.get(where={"user_id": user_id})
                    if res['ids']:
                        col.delete(ids=res['ids'])
            return True
        except Exception as e:
            print(f"Error deleting user data: {str(e)}")
            return False

    def get_core_insight(self, user_id: str) -> str:
        """Retrieves the ultra-concise user life insight (max 20 tokens)"""
        try:
            active_cols = self.collections[self.active_provider]
            results = active_cols["profiles"].get(where={"$and": [{"user_id": user_id}, {"type": "core_insight"}]})
            if results['documents']:
                return results['documents'][0]
            return ""
        except Exception as e:
            print(f"Error getting core insight: {e}")
            return ""

    def update_core_insight(self, user_id: str, insight_text: str):
        """Updates the ultra-concise user life insight"""
        try:
            active_cols = self.collections[self.active_provider]
            # Delete old core insight
            existing = active_cols["profiles"].get(where={"$and": [{"user_id": user_id}, {"type": "core_insight"}]})
            if existing['ids']:
                active_cols["profiles"].delete(ids=existing['ids'])
            
            # Limit to ~20 words just in case (as requested)
            words = insight_text.split()
            if len(words) > 25: # small buffer
                insight_text = " ".join(words[:20]) + "..."
            
            self.store_memory(user_id, "core_insight", insight_text, tags=["profile", "core_insight"])
        except Exception as e:
            print(f"Error updating core insight: {e}")

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Deletes all memories associated with a conversation ID across all collections and providers"""
        try:
            count = 0
            # Common filter for all deletions
            filter_criteria = {"$and": [{"user_id": user_id}, {"conversation_id": conversation_id}]}
            
            for provider in self.collections:
                cols = self.collections[provider]
                for col_name, col in cols.items():
                    try:
                        # Before deleting, count how many for logging
                        results = col.get(where=filter_criteria, include=[])
                        if results and results['ids']:
                            num_to_delete = len(results['ids'])
                            col.delete(where=filter_criteria)
                            count += num_to_delete
                            print(f"[MEMORY] Purged {num_to_delete} items from {provider}/{col_name}")
                    except Exception as col_err:
                        print(f"[MEMORY WARNING] Error purging collection {provider}/{col_name}: {col_err}")
                        
            print(f"[MEMORY] Successfully deleted total {count} items for conversation {conversation_id}")
            return True
        except Exception as e:
            print(f"[MEMORY ERROR] Failed to delete conversation: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --- Conversation History & Threading ---

    def get_conversation_threads(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves a list of conversation threads — searches ALL provider collections."""
        try:
            threads = {}
            
            # Search across every provider's logs collection
            for provider, cols in self.collections.items():
                logs_col = cols["logs"]
                try:
                    results = logs_col.get(where={"user_id": user_id}, limit=10000)
                except Exception:
                    results = None
                if not results or not results.get('ids'):
                    continue
                    
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i]
                    conv_id = meta.get('conversation_id')
                    if not conv_id: continue
                    
                    timestamp = meta.get('timestamp', '')
                    
                    if conv_id not in threads:
                        threads[conv_id] = {
                            "id": conv_id,
                            "last_message": timestamp,
                            "preview": doc[:100],
                            "message_count": 0,
                            "title": f"Session {conv_id[:8]}"
                        }
                    
                    threads[conv_id]["message_count"] += 1
                    if timestamp > threads[conv_id]["last_message"]:
                        threads[conv_id]["last_message"] = timestamp
                        threads[conv_id]["preview"] = doc[:100]
            
            thread_list = list(threads.values())
            
            # Generate readable titles from preview text
            for thread in thread_list:
                if thread["title"].startswith("Session ") and len(thread["title"]) <= 16:
                    words = thread["preview"].replace("User:", "").replace("AI:", "").strip().split()
                    if words:
                        keyword_title = " ".join(words[:5])
                        if len(keyword_title) > 30: keyword_title = keyword_title[:27] + "..."
                        thread["title"] = keyword_title

            thread_list.sort(key=lambda x: x['last_message'], reverse=True)
            return thread_list
            
        except Exception as e:
            print(f"Error getting conversation threads: {e}")
            return []

    def get_conversation_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves flat conversation history — searches ALL provider collections."""
        try:
            history = []
            seen_ids = set()
            for provider, cols in self.collections.items():
                try:
                    results = cols["logs"].get(where={"user_id": user_id}, limit=limit)
                except Exception:
                    continue
                if results and results.get('ids'):
                    for i, doc in enumerate(results['documents']):
                        rid = results['ids'][i]
                        if rid in seen_ids: continue
                        seen_ids.add(rid)
                        history.append({
                            "text": doc,
                            "timestamp": results['metadatas'][i].get('timestamp'),
                            "id": rid
                        })
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []

    def get_conversation_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a conversation - searches ALL providers."""
        try:
            messages = []
            seen_ids = set()
            where_filter = {"$and": [{"user_id": user_id}, {"conversation_id": conversation_id}]}
            for provider, cols in self.collections.items():
                try:
                    results = cols["logs"].get(where=where_filter)
                except Exception:
                    continue
                if results and results.get('ids'):
                    for i, doc in enumerate(results['documents']):
                        rid = results['ids'][i]
                        if rid in seen_ids: continue
                        seen_ids.add(rid)
                        messages.append({"text": doc, "timestamp": results['metadatas'][i].get('timestamp')})
            messages.sort(key=lambda x: x.get('timestamp', ''))
            return messages
        except Exception as e:
            print(f"Error getting conversation messages: {e}")
            return []
# REPLACED_MARKER