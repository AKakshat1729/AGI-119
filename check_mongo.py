import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://abc:1234@cluster0.jlrvd9l.mongodb.net/")

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("SUCCESS: Connected to MongoDB")
    
    db = client['agi-therapist']
    users_collection = db['users']
    
    users = list(users_collection.find())
    print(f"DEBUG: Found {len(users)} users")
    for user in users:
        email = user.get('email')
        settings = user.get('settings', {})
        api_key = settings.get('gemini_api_key')
        print(f"User: {email}, Has Key: {'Yes' if api_key else 'No'}")
        if api_key:
             print(f"Key starts with: {api_key[:10]}...")
except Exception as e:
    print(f"FAILED: {e}")
