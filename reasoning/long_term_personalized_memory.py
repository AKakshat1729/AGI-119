import os
import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, cast
import chromadb
from chromadb.utils import embedding_functions

# --- Fallback Integration ---
try:
    from core.clinical_intelligence import MedicalProfileExtractor
except ImportError:
    MedicalProfileExtractor = None

class PersonalizedMemoryModule:
    def __init__(self, db_path="personalized_memory.db", vector_path="personalized_vector_db"):
        self.db_path = db_path
        self.vector_path = vector_path
        
        # Initialize SQLite for structured data
        self._init_sqlite()
        
        # Initialize ChromaDB for vector data
        # Using the specified local embedding model
        try:
            from chromadb.utils import embedding_functions
            # Cast to Any to prevent Pylance "Contravariance" errors
            raw_emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.emb_fn = cast(Any, raw_emb_fn)
        except Exception as e:
            print(f"[MEMORY] Error loading sentence-transformers: {e}. Falling back to default.")
            self.emb_fn = None

        self.chroma_client = chromadb.PersistentClient(path=self.vector_path)
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="user_memories",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.emb_fn
            )
        except Exception as e:
            # Handle embedding function conflict: if it fails, recreate
            print(f"[MEMORY] Collection initialization failed ({e}), recreating to resolve conflict...")
            try:
                self.chroma_client.delete_collection("user_memories")
            except: pass
            self.collection = self.chroma_client.get_or_create_collection(
                name="user_memories",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.emb_fn
            )

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                category TEXT,
                key TEXT,
                value TEXT,
                importance_score REAL,
                timestamp TEXT,
                ttl_expires TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def store_memory_object(self, user_id: str, memory_obj: Dict[str, Any]):
        """Store a structured memory object in both SQLite and Vector Store"""
        memory_id = f"{user_id}_{datetime.now().timestamp()}_{memory_obj.get('key', 'generic')}"
        timestamp = datetime.now().isoformat()
        
        # 1. Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memories (id, user_id, category, key, value, importance_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_id,
            user_id,
            memory_obj.get('category'),
            memory_obj.get('key'),
            memory_obj.get('value'),
            memory_obj.get('importance_score', 0.5),
            timestamp
        ))
        conn.commit()
        conn.close()

        # 2. Store in Vector Store
        key_id = str(memory_obj.get('key', 'generic')).lower().replace(" ", "_")
        vector_id = f"{user_id}_{key_id}" if memory_obj.get('category') in ['identity', 'medical'] else memory_id
        
        # Ensure document is always a string and never None
        doc_content = str(memory_obj.get('value') or "empty_memory")

        self.collection.upsert(
            documents=[doc_content],
            metadatas=[{
                "user_id": user_id,
                "category": memory_obj.get('category'),
                "key": memory_obj.get('key'),
                "importance": memory_obj.get('importance_score', 0.5),
                "timestamp": timestamp
            }],
            ids=[vector_id]
        )

    def retrieve_relevant_memories(self, user_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search for relevant memories"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": user_id}
            )
            
            memories = []
            
            # --- PYLANCE SAFETY WRAP ---
            # We capture these in variables so Pylance knows they aren't 'None'
            docs_field = results.get('documents')
            ids_field = results.get('ids')
            metas_field = results.get('metadatas')
            dists_field = results.get('distances')

            # 1. Check if the main list exists and has content
            if results and docs_field is not None and len(docs_field) > 0:
                doc_list = docs_field[0]
                
                # 2. Extract others safely
                id_list = ids_field[0] if ids_field is not None else []
                meta_list = metas_field[0] if metas_field is not None else []
                dist_list = dists_field[0] if dists_field is not None else []

                for i in range(len(doc_list)):
                    memories.append({
                        "id": id_list[i] if i < len(id_list) else f"gen_{i}",
                        "value": doc_list[i],
                        "metadata": meta_list[i] if i < len(meta_list) else {},
                        "distance": dist_list[i] if i < len(dist_list) else 0.0
                    })
            
            # Final sort by importance
            memories.sort(key=lambda x: (1 - float(x.get('distance', 0))) * float(x.get('metadata', {}).get('importance', 0.5)), reverse=True)
            return memories

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def get_user_memory_context_formatted(self, user_id: str, query: str) -> str:
        """Get formatted context for LLM prompt"""
        relevant = self.retrieve_relevant_memories(user_id, query)
        if not relevant:
            return ""
        
        context_parts = []
        for mem in relevant:
            context_parts.append(f"- {mem['value']}")
        
        return "\n".join(context_parts)

    def get_full_memory_report(self, user_id: str) -> Dict[str, List]:
        """API Endpoint helper: GET /user-memory-context/{user_id}"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM memories WHERE user_id = ?', (user_id,))
        rows = cursor.fetchall()
        conn.close()

        report: Dict[str, List] = {
            "important_identity": [],
            "medical_flags": [],
            "life_story_events": [],
            "risk_indicators": [],
            "recurring_themes": []
        }

        mapping = {
            "identity": "important_identity",
            "medical": "medical_flags",
            "psychological": "medical_flags",
            "life_story": "life_story_events",
            "risk": "risk_indicators",
            "theme": "recurring_themes"
        }

        for row in rows:
            cat = row['category']
            target_key = mapping.get(cat, "recurring_themes")
            report[target_key].append({
                "key": row['key'],
                "value": row['value'],
                "importance": row['importance_score'],
                "timestamp": row['timestamp']
            })
        
        return report

    def analyze_historical_data(self, user_id: str, conversations: List[Dict], llm_client_func, api_key: Optional[str] = None):
        """Process historical conversations to extract long-term memory."""
        print(f"[MEMORY] Starting historical analysis for {user_id} across {len(conversations)} sessions...")
        
        for i, convo in enumerate(conversations):
            try:
                if "messages" in convo:
                    transcript = ""
                    for msg in convo["messages"]:
                        role = str(msg.get("role", "user"))
                        text = str(msg.get("text", msg.get("content", "")))
                        transcript += f"{role.capitalize()}: {text}\n"
                else:
                    transcript = str(convo.get("transcript", ""))

                if not transcript or len(transcript) < 50:
                    continue
                
                print(f"  -> Analyzing session {i+1}/{len(conversations)}...")
                transcript_trimmed = transcript[:3000] if len(transcript) > 3000 else transcript
                prompt = f"""Analyze this SUSTAINED therapy session transcript and extract critical personal information into structured JSON.
Categories: identity, medical, risk, life_story, theme.
Format as a JSON list of objects: [{{"category": "...", "key": "...", "value": "...", "importance_score": 0.0-1.0}}]

TRANSCRIPT:
{transcript_trimmed}

Extract ALL significantly useful information for long-term therapeutic continuity.
"""
                # Safety wrap api_key to ensure it's a string for the LLM function
                response = llm_client_func([{"role": "user", "content": prompt}], api_key=str(api_key or ""))
                raw_response = str(response.get('response', ''))
                
                try:
                    if "```json" in raw_response:
                        json_str = raw_response.split("```json")[1].split("```")[0].strip()
                    else:
                        json_str = raw_response.strip()
                    
                    memories = json.loads(json_str)
                    if isinstance(memories, list):
                        for mem in memories:
                            self.store_memory_object(user_id, mem)
                       
                except Exception as e:
                    print(f"  [!] Failed to parse session {i+1}: {e}")
                    
            except Exception as e:
                print(f"  [!] Error processing historical session {i}: {e}")
        
        print(f"[MEMORY] Historical analysis completed for {user_id}.")

    def extract_and_save_async(self, user_id: str, transcript: str, llm_client_func, api_key: Optional[str] = None):
        """Asynchronously extract and save memories post-session."""
        def task():
            try:
                # 1. IMMEDIATE LOCAL PASS (Zero API Cost)
                if MedicalProfileExtractor:
                    try:
                        extractor = MedicalProfileExtractor()
                        local_rep = extractor.build_full_report(user_id, [transcript])
                        for risk in local_rep.get('risk_flags', []):
                            self.store_memory_object(user_id, {
                                "category": "risk",
                                "key": "safety_alert",
                                "value": risk,
                                "importance_score": 0.9
                            })
                        id_data = local_rep.get('identity', {})
                        for k, v in id_data.items():
                            if v:
                                self.store_memory_object(user_id, {
                                    "category": "identity",
                                    "key": k,
                                    "value": v,
                                    "importance_score": 0.7
                                })
                    except Exception as le:
                        print(f"[MEMORY] Local pass failed: {le}")

                if not llm_client_func:
                    print("[MEMORY] LLM pass skipped. Local extraction only.")
                    return

                transcript_trimmed = transcript[:3000] if len(transcript) > 3000 else transcript
                prompt = f"""Analyze this therapy session transcript and extract critical personal information into structured JSON.
Categories: identity, medical, risk, life_story, theme.
Format as a JSON list of objects: [{{"category": "...", "key": "...", "value": "...", "importance_score": 0.0-1.0}}]

TRANSCRIPT:
{transcript_trimmed}

Extract ONLY significantly new or updated information. Prioritize risk indicators.
"""
                response = llm_client_func([{"role": "user", "content": prompt}], api_key=str(api_key or ""))
                
                if not response or 'error' in response or not response.get('response'):
                    err = response.get('error', 'Unknown API Error')
                    print(f"[MEMORY] API returned error: {err}")
                    return

                raw_response = str(response.get('response', ''))
                try:
                    if "```json" in raw_response:
                        json_str = raw_response.split("```json")[1].split("```")[0].strip()
                    else:
                        json_str = raw_response.strip()
                    
                    memories = json.loads(json_str)
                    if isinstance(memories, list):
                        for mem in memories:
                            self.store_memory_object(user_id, mem)
                except Exception as e:
                    print(f"[MEMORY] Failed to parse extraction JSON: {e}")
            except Exception as e:
                print(f"[MEMORY] Extraction task failed: {e}")

        threading.Thread(target=task, daemon=True).start()