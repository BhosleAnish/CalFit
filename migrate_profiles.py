import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get Mongo connection details from .env
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["user_profiles"]  # collection you want to insert into

print(f"✅ Connected to MongoDB database: {DB_NAME}")

# Load your CSV
csv_file = "user_profiles.csv"  # make sure this path is correct
df = pd.read_csv(csv_file)

# Convert DataFrame to dicts
records = df.to_dict(orient="records")

# Insert into MongoDB
if records:
    result = collection.insert_many(records)
    print(f"✅ Inserted {len(result.inserted_ids)} documents into user_profiles collection.")
else:
    print("⚠️ No records found in CSV.")
print("✅ Migration completed.")