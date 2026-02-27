import chromadb
from chromadb.config import Settings

class LongTermMemory:
    def __init__(self, user_id="default", collection_name="long_term_memory"):
        """
        Initializes the LongTermMemory with a ChromaDB persistent client and a collection per user.
        Args:
            user_id (str): The ID of the user. Defaults to "default".
            collection_name (str): The name of the collection to use. Defaults to "long_term_memory".
        """
        self.user_id = user_id
        self.client = chromadb.PersistentClient(path=f"./long_term_memory_db_{user_id}")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, knowledge, id=None):
        """
        Stores knowledge in the long term memory.
        Args:
            knowledge (str): The knowledge to store.
            id (str, optional): The ID of the data. Defaults to None.
        """
        # knowledge as string
        text = str(knowledge)
        if id is None:
            id = str(len(self.collection.get()['ids']) + 1)
        self.collection.add(documents=[text], ids=[id])

    def retrieve(self, query, n_results=10):
        """
        Retrieves data from the long term memory based on a query.
        Args:
            query (str): The query to use.
            n_results (int, optional): The number of results to return. Defaults to 10.
        Returns:
            list: The results of the query.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results

    def update(self, id, new_knowledge):
        """
        Updates knowledge in the long term memory.
        Args:
            id (str): The ID of the data to update.
            new_knowledge (str): The new knowledge to store.
        """
        # ChromaDB doesn't support direct update, so delete and add
        self.collection.delete(ids=[id])
        self.store(new_knowledge, id)

    def get_all(self):
        """
        Retrieves all stored information from the collection for display.
        Returns:
            dict: A dictionary containing all 'ids', 'documents', and 'metadatas'.
        """
        return self.collection.get()
