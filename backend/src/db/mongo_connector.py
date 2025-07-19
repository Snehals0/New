# backend/src/db/mongo_connector.py
from pymongo import MongoClient
from backend.config.db_config import MONGO_URI, MONGO_DB_NAME
import time # For retry logic

def get_mongo_client(retries=10, delay=3):
    """Attempts to establish a MongoDB connection with retries."""
    for i in range(retries):
        try:
            client = MongoClient(MONGO_URI)
            # The next line will try to connect and throw an exception if fails
            client.admin.command('ping') # Test connection
            print(f"MongoDB connection successful on attempt {i+1}.")
            return client[MONGO_DB_NAME] # Returns the database object
        except Exception as e:
            print(f"MongoDB connection attempt {i+1}/{retries} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                print("Max retries reached. Could not connect to MongoDB.")
                return None
    return None