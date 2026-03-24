import pymongo, os
from dotenv import load_dotenv
load_dotenv()
uri = os.getenv('MONGO_URI')
print('URI found:', bool(uri))
client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
print('Connected!')