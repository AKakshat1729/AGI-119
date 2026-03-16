import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    mongo_uri = os.environ.get("MONGO_URI")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client
