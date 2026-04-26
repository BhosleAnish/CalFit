import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

class MongoConfig:
    def __init__(self):
        self.MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.DATABASE_NAME = os.getenv('MONGO_DB_NAME', 'nutrition_app')

        try:
            self.client = pymongo.MongoClient(
                self.MONGO_URI,
                retryWrites=True,
                w='majority',
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            self.db = self.client[self.DATABASE_NAME]
            self.client.admin.command('ping')
            print("✅ MongoDB connection successful")
            print(f"✅ Connected to database: {self.DATABASE_NAME}")

        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"❌ MongoDB connection timeout: {e}")
            self.client = None
            self.db = None
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None