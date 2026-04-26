from datetime import datetime
from .database import MongoConfig


class UserProfileStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_profiles']
            self._setup_indexes()
            print("✅ User profiles collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user profiles collection: {e}")
            self.collection = None

    def _setup_indexes(self):
        try:
            self.collection.create_index("username", unique=True)
            self.collection.create_index("created_at")
            print("✅ User profile indexes created")
        except Exception as e:
            print(f"⚠️ Could not create profile indexes: {e}")

    def save_profile(self, profile_data):
        """Insert or update a user profile."""
        if self.collection is None:
            return False

        username = profile_data.get('username')
        if not username:
            print("❌ Username required to save profile")
            return False

        try:
            doc = {
                "username": username,
                "full_name":          profile_data.get('full_name', ''),
                "email":              profile_data.get('email', ''),
                "age":                int(profile_data['age']) if profile_data.get('age') else None,
                "gender":             profile_data.get('gender', ''),
                "height_cm":          float(profile_data['height_cm']) if profile_data.get('height_cm') else None,
                "weight_kg":          float(profile_data['weight_kg']) if profile_data.get('weight_kg') else None,
                "activity_level":     profile_data.get('activity_level', ''),
                "medical_conditions": profile_data.get('medical_conditions', ''),
                "allergies":          profile_data.get('allergies', ''),
                "updated_at":         datetime.utcnow(),
            }

            existing = self.collection.find_one({"username": username})
            if existing:
                doc['created_at'] = existing.get('created_at', datetime.utcnow())
                result = self.collection.update_one({"username": username}, {"$set": doc})
                print(f"✅ Profile updated: {username}")
                return result.matched_count > 0
            else:
                doc['created_at'] = datetime.utcnow()
                result = self.collection.insert_one(doc)
                print(f"✅ Profile created: {username}")
                return result.inserted_id is not None

        except Exception as e:
            print(f"❌ Failed to save profile: {e}")
            return False

    def get_profile(self, username):
        """Return the profile dict for a user, or None."""
        if self.collection is None:
            return None
        try:
            profile = self.collection.find_one({"username": username})
            if profile:
                profile.pop('_id', None)
            return profile
        except Exception as e:
            print(f"❌ Failed to retrieve profile: {e}")
            return None

    def profile_exists(self, username):
        """Return True if a profile exists for this username."""
        if self.collection is None:
            return False
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            print(f"❌ Failed to check profile existence: {e}")
            return False

    def delete_profile(self, username):
        """Delete the profile for a user."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Failed to delete profile: {e}")
            return False