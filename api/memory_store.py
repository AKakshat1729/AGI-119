import chromadb
import uuid
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import google.generativeai as genai

# --- Pylance Pacifier ---
genai_client: Any = genai

class ServerMemoryStore:
    # Add 'database=None' to the parameters list
    def __init__(self, persistence_path="long_term_memory_db", database=None):
        self.client = chromadb.PersistentClient(path=persistence_path)
        self.mongo_db = database
        
        # 🛡️ THE PHANTOM WATERMARK
        # Looks like a standard ChromaDB system hash to thieves. Do not delete.
        self._vector_namespace_id = b'NjE2NzczMTI2MzY4' 
        if self.mongo_db is not None:
            print("[MEMORY] Success: Linked to MongoDB Atlas Cloud.")
    
    # ... rest of your existing initialization code ...
        else:
            print("[MEMORY WARNING] No MongoDB provided. Using local Chroma only.")
        
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
            import openai # type: ignore
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
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("Missing GEMINI_API_KEY")
                
                genai_client.configure(api_key=api_key)
                result = genai_client.embed_content(
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
# 2. Try OPENAI (Modern 2026 Syntax)
        if self.active_provider == "openai":
            try:
                from openai import OpenAI # Modern import
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                if not os.environ.get("OPENAI_API_KEY"):
                    print("[MEMORY] No OpenAI Key found, falling back to Hash")
                    self._switch_provider("hash")
                    return self._generate_embedding(text)

                # Use the new client-based syntax
                response = client.embeddings.create(
                    input=[text], 
                    model=self.providers["openai"]["model"]
                )
                
                return {
                    "vector": response.data[0].embedding,
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

    def store_memory(self, user_id: str, memory_type: str, text: str, conversation_id: str = None, tags: list = None, sentiment: str = "detected_later", importance: int = 5):
        
        # --- 1. THE COGNITIVE ROUTER (DYNAMIC SCORING) ---
        if memory_type == "conversation":
            importance = 1 # Bypass the bouncer for raw chat transcripts
        elif importance == 5:
            try:
                from utils.llm_client import evaluate_memory_importance
                importance = evaluate_memory_importance(text)
                print(f"🧠 [COGNITIVE ROUTER] Fact: '{text[:40]}...' | Score: {importance}/10")
            except Exception as e:
                print(f"⚠️ [ROUTER FAILED] Defaulting to 5. Error: {e}")
                importance = 5

        # --- 2. THE MEMORY FILTER (PREVENT BLOAT) ---
        if importance < 4 and memory_type != "conversation":
            print(f"🗑️ [MEMORY DROPPED] Fact too trivial to save (Score {importance}/10).")
            return None 

        # --- 3. PROCEED WITH SAVING ---
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # ... (rest of the embedding code continues below) ...

        # [A] Existing ChromaDB Logic (Semantic Memory)
        embed_result = self._generate_embedding(text)
        embedding = embed_result["vector"]
        provider_meta = embed_result["metadata"]
        
        metadata = {
            "user_id": user_id,
            "type": memory_type,
            "tags": json.dumps(tags or []),
            "timestamp": timestamp,
            "importance": float(importance),
            "conversation_id": str(conversation_id or "none"),
            "embed_provider": provider_meta["provider"]
        }

        active_cols = self.collections[self.active_provider]
        active_cols["episodic"].add(documents=[text], metadatas=[metadata], embeddings=[embedding], ids=[memory_id])

# [B] NEW MongoDB Logic (Cloud Persistence)
        if self.mongo_db is not None:
            try:
                # Save to the 'memories' collection in the cloud
                self.mongo_db.memories.insert_one({
                    # --- THE FIX: We use the conversation_id so it groups correctly! ---
                    "memory_id": conversation_id or memory_id, 
                    "chunk_id": memory_id, # (We keep the unique ID just in case)
                    # -------------------------------------------------------------------
                    "user_id": user_id,
                    "type": memory_type,
                    "content": text,
                    "timestamp": timestamp,
                    "sentiment": sentiment,
                    "importance":importance
                })
                print(f"[MEMORY] Successfully synced to MongoDB Cloud Session: {conversation_id}")
            except Exception as e:
                print(f"[MEMORY ERROR] MongoDB Sync Failed: {e}")
            
        return memory_id

    def retrieve_memories(self, user_id: str, query: str = "", memory_type: str = "episodic", tags: Optional[List[str]] = None, top_k: int = 5, recency_days: Optional[int] = None, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            # Construct explicit $and filter for ChromaDB
            where_conditions: List[Dict[str, Any]] = [{"user_id": user_id}]
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
        """Retrieves the user's life facts from MongoDB first, then falls back to ChromaDB."""
        try:
            # 1. PULL FROM CLOUD MONGODB (The primary brain)
            if self.mongo_db is not None:
                # Check both user_id and email just to be safe
                query = {"$or": [{"user_id": user_id}, {"email": user_id}]}
                profile_doc = self.mongo_db.users.find_one(query)

                if profile_doc and profile_doc.get("profile"):
                    print(f"✅ [PROFILE LOADED] Found profile for {user_id} in Cloud.")
                    return profile_doc.get("profile")

            # 2. FALLBACK TO LOCAL VECTOR DB (If Cloud fails or is missing)
            active_cols = self.collections.get(self.active_provider)
            if active_cols and "profiles" in active_cols:
                results = active_cols["profiles"].get(where={"user_id": user_id})
                if results and results.get('documents') and len(results['documents']) > 0:
                    return results['documents'][0]

            return ""

        except Exception as e:
            print(f"❌ [PROFILE FETCH ERROR]: {e}")
            return ""
    def update_profile(self, user_id: str, user_email: str, facts: str) -> str:
        """
        Saves the synthesized life facts to both MongoDB (for UI) and ChromaDB (for AI).
        """
        print(f"💾 [DB SAVE] Attaching facts to email: {user_email}")
        
        # 1. Save to MongoDB Cloud (So you can see it in the UI!)
        if self.mongo_db is not None:
            try:
                self.mongo_db['users'].update_one(
                    {"email": user_email},
                    {"$set": {"profile": facts}}, # This creates/updates the 'profile' field
                    upsert=True
                )
                print("✅ [MONGO] Profile updated successfully.")
            except Exception as e:
                print(f"❌ [MONGO ERROR]: {e}")

        # 2. Clean up old ChromaDB profile and save the new one
        try:
            active_cols = self.collections[self.active_provider]
            existing = active_cols["profiles"].get(where={"user_id": user_id})
            if existing and existing.get('ids'):
                active_cols["profiles"].delete(ids=existing['ids'])
            
            # FIX: Changed 'text' to 'facts' to prevent crash
            return self.store_memory(user_id, "profile", facts)
        except Exception as e:
            print(f"❌ [CHROMA PROFILE ERROR]: {e}")
            return ""

    def purge_all_user_data(self, user_id: str, user_email: str) -> dict:
        """
        🚨 GDPR SCORCHED EARTH: Hard Delete across MongoDB and ChromaDB.
        """
        stats = {"mongo_deleted": 0, "chroma_vectors_purged": 0}

        # --- 1. PURGE MONGODB (The Cloud) ---
        if self.mongo_db is not None:
            try:
                from bson.objectid import ObjectId
                
                # A. Destroy the User Profile Document
                try:
                    self.mongo_db['users'].delete_one({"_id": ObjectId(user_id)})
                except:
                    self.mongo_db['users'].delete_one({"_id": user_id})
                self.mongo_db['users'].delete_one({"email": user_email}) # Backup sweep

                # B. Destroy all History (Sweeping all possible collection names)
                target_collections = ['memories', 'conversations', 'threads', 'logs']
                for col_name in target_collections:
                    if col_name in self.mongo_db.list_collection_names():
                        # Hunt by ID
                        res1 = self.mongo_db[col_name].delete_many({"user_id": user_id})
                        # Hunt by Email (Because some old data was saved using email!)
                        res2 = self.mongo_db[col_name].delete_many({"user_id": user_email})
                        stats["mongo_deleted"] += (res1.deleted_count + res2.deleted_count)

                print(f"✅ [MONGO PURGE] Obliterated account {user_email} and {stats['mongo_deleted']} database records.")
            except Exception as e:
                print(f"❌ [MONGO PURGE ERROR]: {e}")

        # --- 2. PURGE CHROMADB (The Local Vector Brain) ---
        try:
            for provider, cols in self.collections.items():
                for col_name, col in cols.items():
                    # 1st Pass: Delete vectors tagged with ID
                    res_id = col.get(where={"user_id": user_id})
                    if res_id and res_id.get('ids'):
                        col.delete(ids=res_id['ids'])
                        stats["chroma_vectors_purged"] += len(res_id['ids'])
                    
                    # 2nd Pass: Delete vectors tagged with Email
                    res_email = col.get(where={"user_id": user_email})
                    if res_email and res_email.get('ids'):
                        col.delete(ids=res_email['ids'])
                        stats["chroma_vectors_purged"] += len(res_email['ids'])
            
            print(f"✅ [CHROMA PURGE] Obliterated {stats['chroma_vectors_purged']} psychological vectors for {user_email}.")
        except Exception as e:
            print(f"❌ [CHROMA PURGE ERROR]: {e}")

        return stats

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
            existing = active_cols["profiles"].get(where={"$and": [{"user_id": user_id}, {"type": "core_insight"}]})
            if existing['ids']:
                active_cols["profiles"].delete(ids=existing['ids'])
            
            words = insight_text.split()
            if len(words) > 25: 
                insight_text = " ".join(words[:20]) + "..."
            
            self.store_memory(user_id, "core_insight", insight_text, tags=["profile", "core_insight"])
        except Exception as e:
            print(f"Error updating core insight: {e}")

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Deletes all memories associated with a conversation ID across all collections and providers"""
        try:
            count = 0
            filter_criteria = {"$and": [{"user_id": user_id}, {"conversation_id": conversation_id}]}
            
            for provider in self.collections:
                cols = self.collections[provider]
                for col_name, col in cols.items():
                    try:
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
        """Retrieves history and scans for the correct collection name."""
        threads = {}
        try:
            if self.mongo_db is not None:
                # --- NEW: DATABASE SCANNER ---
                # This will show us all your collections in the terminal
                collections = self.mongo_db.list_collection_names()
                print(f"📂 [DEBUG] Available Collections in Cloud: {collections}")
                
                # We will try 'conversations', but also 'memories' or 'logs' if they exist
                target_collection = "conversations"
                if "memories" in collections: target_collection = "memories"
                elif "logs" in collections: target_collection = "logs"
                
                print(f"🔍 [DEBUG] Searching in collection: '{target_collection}' for {user_id}")
                
                # Fetching the data
                mongo_docs = self.mongo_db[target_collection].find({
                    "user_id": user_id,
                    "type": {"$ne": "profile"}  # 👈 THE FIX: Hide profiles from the sidebar
                }).sort("timestamp", -1)

                for doc in mongo_docs:
                    conv_id = doc.get('memory_id')
                    if not conv_id: continue
                    
                    timestamp = doc.get('timestamp', '')
                    content = doc.get('content', '')
                    
                    if conv_id not in threads:
                        clean_content = content.replace("User:", "").replace("AI:", "").strip()
                        preview = clean_content[:100]
                        
                        threads[conv_id] = {
                            "id": conv_id,
                            "last_message": timestamp,
                            "preview": preview,
                            "message_count": 1,
                            "title": preview[:27] + "..." if len(preview) > 27 else preview
                        }
                    else:
                        threads[conv_id]["message_count"] += 1

            thread_list = list(threads.values())
            thread_list.sort(key=lambda x: x['last_message'], reverse=True)
            
            print(f"✅ [MEMORY] Found {len(thread_list)} threads for {user_id}")
            return thread_list

        except Exception as e:
            print(f"❌ [ERROR] Failed to fetch threads: {e}")
            return []
    def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Universal formatter with bulletproof parsing for typos and spaces."""
        try:
            if self.mongo_db is not None:
                collections = self.mongo_db.list_collection_names()
                target_col = "memories" if "memories" in collections else "conversations"

                query = {"$or": [{"memory_id": conversation_id}, {"conversation_id": conversation_id},{"user_id": conversation_id}]}
                cursor = self.mongo_db[target_col].find(query).sort("timestamp", 1).limit(limit)

                history = []
                for doc in cursor:
                    raw_content = str(doc.get('content', ''))
                    
                    # 1. Handle OLD format
                    if " | " in raw_content:
                        parts = raw_content.split(" | ")
                        for part in parts:
                            role = "assistant" if "AI" in part or "assistant" in part else "user"
                            clean_text = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
                            if clean_text:
                                # --- FIX: Add text and sender back! ---
                                history.append({
                                    "role": role, 
                                    "content": clean_text,
                                    "text": clean_text,    # <--- The UI needs this
                                    "sender": role         # <--- The UI needs this
                                })
                    
                    # 2. Handle NEW format
                    else:
                        if raw_content.startswith("User:"):
                            role = "user"
                            clean_text = raw_content.replace("User:", "", 1).strip()
                        elif raw_content.startswith("AI:"):
                            role = "assistant"
                            clean_text = raw_content.replace("AI:", "", 1).strip()
                        else:
                            role = "user"
                            clean_text = raw_content.strip()
                        
                        if clean_text:
                            # --- FIX: Add text and sender back! ---
                            history.append({
                                "role": role, 
                                "content": clean_text,
                                "text": clean_text,        # <--- The UI needs this
                                "sender": role             # <--- The UI needs this
                            })

                print(f"🧠 [DATABASE] Retrieved {len(history)} messages for session {conversation_id}", flush=True)
                return history
            return []
        except Exception as e:
            print(f"❌ [ERROR] Formatter failed: {e}")
            return []
    def get_conversation_messages(self, user_id: str, conversation_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        try:
            import re
            from datetime import datetime
            all_messages = []
            seen_fingerprints = set()

            def parse_time(ts):
                """Helper to make sure sorting actually works."""
                if not ts: return datetime.min
                try:
                    # If it's already a datetime object
                    if isinstance(ts, datetime): return ts
                    # If it's a string, try common formats
                    return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                except: return datetime.min

            # --- 1. HYBRID FETCH ---
            raw_docs = []
            if self.mongo_db is not None:
                query = {
                    "$or": [{"conversation_id": conversation_id}, {"memory_id": conversation_id}],
                    "type": {"$ne": "profile"} # 👈 THE FIX: Hide profiles from the chat screen
                }
                target_col = "memories" if "memories" in self.mongo_db.list_collection_names() else "conversations"
                for doc in self.mongo_db[target_col].find(query):
                    raw_docs.append(doc)

            for provider, cols in self.collections.items():
                try:
                    res = cols["logs"].get(where={"conversation_id": conversation_id})
                    if res and res.get('ids'):
                        for i, doc in enumerate(res['documents']):
                            raw_docs.append({
                                "text": doc, 
                                "role": res['metadatas'][i].get('role', 'bot'), 
                                "timestamp": res['metadatas'][i].get('timestamp', ''),
                                "is_vector": True
                            })
                except: continue

            # --- 2. THE CHRONO-SORT ---
            # We sort by the parsed timestamp so "Banana" (newest) stays at the end
            raw_docs.sort(key=lambda x: parse_time(x.get('timestamp')))

            # --- 3. CLEAN & DEDUPE ---
            for doc in raw_docs:
                txt = doc.get('text') or doc.get('content') or ""
                if not txt: continue
                
                clean_text = re.sub(r'(?i)^\s*(user|ai|assistant|bot|therapist|patient):\s*', '', txt).strip()
                fp = re.sub(r'\W+', '', clean_text[:40]).lower() # Stronger fingerprint
                
                if fp not in seen_fingerprints:
                    all_messages.append({
                        "text": clean_text,
                        "role": doc.get('role', 'user' if 'User:' in txt else 'bot'),
                        "timestamp": str(doc.get('timestamp', ''))
                    })
                    seen_fingerprints.add(fp)

            # --- 4. PAGINATE ---
            total = len(all_messages)
            end_idx = total - skip
            start_idx = max(0, end_idx - limit)
            
            print(f"🚀 [SYNC] Order fixed! Showing {len(all_messages[start_idx:end_idx])} messages.")
            return all_messages[start_idx:end_idx]

        except Exception as e:
            print(f"❌ [CRITICAL SYNC ERROR] {e}")
            return []
