from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from .database import MongoConfig


class ScanStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_scans']
            self._setup_indexes()
            print("✅ User scans collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user scans collection: {e}")
            self.collection = None

    def _setup_indexes(self):
        try:
            self.collection.create_index("username")
            self.collection.create_index("scan_date")
            self.collection.create_index([("username", 1), ("scan_date", -1)])
            self.collection.create_index("expires_at", expireAfterSeconds=0)
            print("✅ Scan indexes created")
        except Exception as e:
            print(f"⚠️ Could not create scan indexes: {e}")

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def save_scan_data(
        self,
        username,
        raw_text,
        cleaned_text,
        structured_nutrients,
        image_filename=None,
        product_name=None,
        product_image_url=None,
    ):
        """Persist a scan document and return its string ID, or None on failure."""
        if self.collection is None:
            return None
        try:
            doc = {
                "username":    username,
                "scan_date":   datetime.utcnow(),
                "product_info": {
                    "name":      product_name or "Scanned Item",
                    "image_url": product_image_url,
                },
                "scan_metadata": {
                    "image_filename":        image_filename,
                    "nutrients_count":       len(structured_nutrients) if structured_nutrients else 0,
                    "processing_timestamp":  datetime.utcnow().isoformat(),
                },
                "ocr_data": {
                    "raw_text":     raw_text,
                    "cleaned_text": cleaned_text,
                },
                "nutrition_analysis": {
                    "structured_nutrients": structured_nutrients,
                    "summary":              self._make_summary(structured_nutrients),
                },
                "system_info": {
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow() + timedelta(days=30),
                    "app_version": "1.0",
                },
            }
            result = self.collection.insert_one(doc)
            print(f"✅ Scan saved: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to save scan: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_user_scans(self, username, limit=50):
        """Return a list of scan dicts for the user, newest first."""
        if self.collection is None:
            return []
        try:
            scans = list(
                self.collection.find({"username": username})
                .sort("scan_date", -1)
                .limit(limit)
            )
            for s in scans:
                s['_id'] = str(s['_id'])
            return scans
        except Exception as e:
            print(f"❌ Failed to retrieve user scans: {e}")
            return []

    def get_scan_by_id(self, scan_id):
        """Return a single scan dict by its MongoDB ObjectId string."""
        if self.collection is None:
            return None
        try:
            scan = self.collection.find_one({"_id": ObjectId(scan_id)})
            if scan:
                scan['_id'] = str(scan['_id'])
            return scan
        except Exception as e:
            print(f"❌ Failed to retrieve scan by ID: {e}")
            return None

    def get_user_scan_stats(self, username):
        """Return summary statistics for a user's scans."""
        empty = {"total_scans": 0, "recent_scans": 0, "latest_scan_date": None, "has_scans": False}
        if self.collection is None:
            return empty
        try:
            total = self.collection.count_documents({"username": username})
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent = self.collection.count_documents({
                "username": username,
                "scan_date": {"$gte": week_ago},
            })
            latest = self.collection.find_one(
                {"username": username}, sort=[("scan_date", -1)]
            )
            return {
                "total_scans":       total,
                "recent_scans":      recent,
                "latest_scan_date":  latest['scan_date'] if latest else None,
                "has_scans":         total > 0,
            }
        except Exception as e:
            print(f"❌ Failed to get scan stats: {e}")
            return empty

    # ------------------------------------------------------------------ #
    #  Delete                                                              #
    # ------------------------------------------------------------------ #

    def delete_scan(self, scan_id, username):
        """Delete a scan only if it belongs to the given user."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({
                "_id": ObjectId(scan_id),
                "username": username,
            })
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Failed to delete scan: {e}")
            return False

    def delete_user_scans(self, username):
        """Delete ALL scans for a user (used when deleting an account)."""
        if self.collection is None:
            return 0
        try:
            result = self.collection.delete_many({"username": username})
            return result.deleted_count
        except Exception as e:
            print(f"❌ Failed to delete user scans: {e}")
            return 0

    def cleanup_expired_scans(self):
        """Manually remove expired scans (TTL index handles this automatically)."""
        if self.collection is None:
            return 0
        try:
            result = self.collection.delete_many({"expires_at": {"$lt": datetime.utcnow()}})
            print(f"🧹 Cleaned up {result.deleted_count} expired scans")
            return result.deleted_count
        except Exception as e:
            print(f"❌ Failed to cleanup expired scans: {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_summary(self, structured_nutrients):
        """Build a compact summary dict from a list of nutrient dicts."""
        if not structured_nutrients:
            return {}
        statuses = [n.get('status') for n in structured_nutrients]
        categories = {}
        for n in structured_nutrients:
            cat = n.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_nutrients": len(structured_nutrients),
            "high_risk_count":  statuses.count('High'),
            "low_risk_count":   statuses.count('Low'),
            "normal_count":     statuses.count('Normal'),
            "unknown_count":    statuses.count('Unknown'),
            "categories":       categories,
        }