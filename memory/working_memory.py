import chromadb
from chromadb.config import Settings

class WorkingMemory:
    def __init__(self, collection_name="working_memory"):
        """
        Initializes the WorkingMemory with a ChromaDB client and a collection.
        Args:
            collection_name (str): The name of the collection to use. Defaults to "working_memory".
        """
        self.client = chromadb.Client(Settings())
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
            id = str(len(self.collection.get()['ids']) + 1)
        self.collection.add(documents=[text], ids=[id])

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

    def clear(self):
        """
        Clears the working memory by recreating the collection.
        """
        # To clear, recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
