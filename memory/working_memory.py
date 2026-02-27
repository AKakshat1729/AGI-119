import chromadb
from chromadb.config import Settings
from datetime import datetime

# Global client to avoid re-initializing on every request
_global_client = None

class WorkingMemory:
    def __init__(self, collection_name="working_memory"):
        """
        Initializes the WorkingMemory with a ChromaDB persistent client and a collection.
        Args:
            collection_name (str): The name of the collection to use. Defaults to "working_memory".
        """
        global _global_client
        if _global_client is None:
            _global_client = chromadb.PersistentClient(path="./working_memory_db")
        
        self.client = _global_client
        # Do NOT pass embedding_function — use ChromaDB default to avoid conflicts
        # with collections that were previously created without a custom embedding fn
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, nlu_output, id=None):
        """
        Stores the NLU output in the working memory.
        Args:
            nlu_output (dict): The NLU output to store.
            id (str, optional): The ID of the data. Defaults to None.
        """
        # Convert nlu_output to string for embedding
        text = str(nlu_output)
        if id is None:
            # Generate a sortable ID using timestamp + incremental counter or just robust UUID
            # But to keep compatibility with existing alphanumeric logic in app.py (which tries to cast to int),
            # we will stick to the existing ID logic but add a timestamp metadata for sorting.
            id = str(len(self.collection.get()['ids']) + 1)
        
        # Add timestamp to metadata
        metadata = {"timestamp": datetime.now().isoformat()}
        
        self.collection.add(documents=[text], metadatas=[metadata], ids=[id])

    def retrieve(self, query, n_results=5):
        """
        Retrieves data from the working memory based on a query.
        Args:
            query (str): The query to use.
            n_results (int, optional): The number of results to return. Defaults to 5.
        Returns:
            list: The results of the query.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results

    def get_all_sorted(self):
        """
        Retrieves all documents sorted chronologically.
        """
        try:
            data = self.collection.get()
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            ids = data.get("ids", [])
            
            # Combine into a list of dicts
            combined = []
            for i in range(len(ids)):
                timestamp = ""
                if metas and i < len(metas) and metas[i]:
                    timestamp = metas[i].get("timestamp", "")
                
                combined.append({
                    "id": ids[i],
                    "document": docs[i],
                    "timestamp": timestamp
                })
            
            # Sort by timestamp if available, else by ID (as int if possible)
            def sort_key(item):
                if item["timestamp"]:
                    return item["timestamp"]
                # Fallback to ID sorting
                try:
                    return int(item["id"])
                except:
                    return item["id"]

            combined.sort(key=sort_key)
            return [item["document"] for item in combined]
        except Exception as e:
            print(f"Error getting sorted working memory: {e}")
            return []

    def clear(self):
        """
        Clears the working memory by recreating the collection.
        """
        # To clear, recreate collection
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
