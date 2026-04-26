import pymongo
from datetime import datetime
from werkzeug.security import generate_password_hash
from .database import MongoConfig


class UserCredentialsStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_credentials']
            self._setup_indexes()
            print("✅ User credentials collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user credentials collection: {e}")
            self.collection = None

    def _setup_indexes(self):
        try:
            self.collection.create_index("username", unique=True)
            print("✅ User credentials indexes created")
        except Exception as e:
            print(f"⚠️ Could not create credentials indexes: {e}")

    def add_user(self, username, password):
        """Add a new regular (non-OAuth) user."""
        if self.collection is None:
            return False
        try:
            self.collection.insert_one({
                "username": username,
                "password": generate_password_hash(password),
                "auth_provider": "regular",
                "created_at": datetime.utcnow()
            })
            print(f"✅ User created: {username}")
            return True
        except pymongo.errors.DuplicateKeyError:
            print(f"⚠️ User already exists: {username}")
            return False
        except Exception as e:
            print(f"❌ Failed to add user: {e}")
            return False

    def get_user(self, username):
        """Get a user document by username."""
        if self.collection is None:
            return None
        try:
            return self.collection.find_one({"username": username})
        except Exception as e:
            print(f"❌ Failed to retrieve user: {e}")
            return None

    def get_user_by_google_id(self, google_id):
        """Get a user document by Google OAuth ID."""
        if self.collection is None:
            return None
        try:
            return self.collection.find_one({"google_id": google_id})
        except Exception as e:
            print(f"❌ Failed to retrieve user by Google ID: {e}")
            return None

    def user_exists(self, username):
        """Return True if a user with this username exists."""
        if self.collection is None:
            return False
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            print(f"❌ Failed to check user existence: {e}")
            return False

    def get_all_users(self):
        """Return a dict of {username: hashed_password} for all users."""
        if self.collection is None:
            return {}
        try:
            return {
                u['username']: u['password']
                for u in self.collection.find({}, {'username': 1, 'password': 1, '_id': 0})
            }
        except Exception as e:
            print(f"❌ Failed to retrieve all users: {e}")
            return {}

    def update_last_login(self, username):
        """Stamp the last_login field for a user."""
        if self.collection is None:
            return
        try:
            self.collection.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        except Exception as e:
            print(f"❌ Failed to update last login: {e}")

    def delete_user(self, username):
        """Delete a user document by username."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Failed to delete user: {e}")
            return False