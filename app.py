from flask import Flask, request, redirect, url_for, session, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os, re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ocr_utils import process_label_image, extract_nutrients
from PIL import Image
import pillow_avif
from io import BytesIO
import sqlite3
import pymongo
from bson import ObjectId
from bson.errors import InvalidId
from process_label import process_nutrition_label
import openai
import json
from flask_cors import CORS
from flask_session import Session
from nlp_analyzer import analyze_report_text
from authlib.integrations.flask_client import OAuth
from functools import wraps

load_dotenv()

# ==================== AUTHENTICATION DECORATOR ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({"error": "Authentication required", "redirect": "/"}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- CONFIGURE THE AI MODEL ---
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client configured successfully.")
    else:
        client = None
        print("⚠️ OPENAI_API_KEY not found in .env file. AI features will be disabled.")
except Exception as e:
    print(f"⚠️ Could not configure OpenAI client: {e}")
    client = None

# ==================== APP INIT ====================
app = Flask(__name__, static_folder="frontend/dist", static_url_path="")

app.secret_key = os.getenv("SECRET_KEY", "super_secret_static_key_123")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_DIR"] = "./.flask_session/"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

Session(app)

# OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# CORS — allow React dev server
CORS(app,
     origins=["http://localhost:5173"],
     supports_credentials=True)

# CSRF — disabled for API (React handles its own security via credentials/cors)
# csrf = CSRFProtect(app)  # Removed: not needed for REST API

# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://"
)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==================== MONGODB ====================
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
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None


class UserCredentialsStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_credentials']
            self.setup_indexes()
            print("✅ User credentials collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user credentials collection: {e}")
            self.collection = None

    def setup_indexes(self):
        try:
            self.collection.create_index("username", unique=True)
            print("✅ User credentials indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create credentials indexes: {e}")

    def add_user(self, username, password):
        if self.collection is None:
            return False
        try:
            self.collection.insert_one({
                "username": username,
                "password": generate_password_hash(password),
                "created_at": datetime.utcnow()
            })
            return True
        except pymongo.errors.DuplicateKeyError:
            return False
        except Exception as e:
            print(f"❌ Failed to add user: {e}")
            return False

    def get_user(self, username):
        if self.collection is None:
            return None
        try:
            return self.collection.find_one({"username": username})
        except Exception as e:
            print(f"❌ Failed to retrieve user: {e}")
            return None

    def user_exists(self, username):
        if self.collection is None:
            return False
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            return False

    def get_all_users(self):
        if self.collection is None:
            return {}
        try:
            users = {}
            for user in self.collection.find({}, {'username': 1, 'password': 1, '_id': 0}):
                users[user['username']] = user['password']
            return users
        except Exception as e:
            return {}

    def get_user_by_google_id(self, google_id):
        if self.collection is None:
            return None
        try:
            return self.collection.find_one({"google_id": google_id})
        except Exception as e:
            return None


class UserProfileStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_profiles']
            self.setup_indexes()
            print("✅ User profiles collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user profiles collection: {e}")
            self.collection = None

    def setup_indexes(self):
        try:
            self.collection.create_index("username", unique=True)
            self.collection.create_index("created_at")
            print("✅ User profile indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create profile indexes: {e}")

    def save_profile(self, profile_data):
        if self.collection is None:
            return False
        try:
            username = profile_data.get('username')
            if not username:
                return False
            profile_document = {
                "username": username,
                "full_name": profile_data.get('full_name', ''),
                "age": int(profile_data.get('age', 0)) if profile_data.get('age') else None,
                "gender": profile_data.get('gender', ''),
                "height_cm": float(profile_data.get('height_cm', 0)) if profile_data.get('height_cm') else None,
                "weight_kg": float(profile_data.get('weight_kg', 0)) if profile_data.get('weight_kg') else None,
                "activity_level": profile_data.get('activity_level', ''),
                "medical_conditions": profile_data.get('medical_conditions', ''),
                "allergies": profile_data.get('allergies', ''),
                "updated_at": datetime.utcnow()
            }
            existing = self.collection.find_one({"username": username})
            if existing:
                profile_document['created_at'] = existing.get('created_at', datetime.utcnow())
                result = self.collection.update_one({"username": username}, {"$set": profile_document})
                return result.modified_count > 0 or result.matched_count > 0
            else:
                profile_document['created_at'] = datetime.utcnow()
                result = self.collection.insert_one(profile_document)
                return result.inserted_id is not None
        except Exception as e:
            print(f"❌ Failed to save profile: {e}")
            return False

    def get_profile(self, username):
        if self.collection is None:
            return None
        try:
            profile = self.collection.find_one({"username": username})
            if profile:
                profile.pop('_id', None)
            return profile
        except Exception as e:
            return None

    def delete_profile(self, username):
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            return False

    def profile_exists(self, username):
        if self.collection is None:
            return False
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            return False


class ScanStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        try:
            self.collection = self.mongo_config.db['user_scans']
            self.setup_indexes()
            print("✅ User scans collection initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user scans collection: {e}")
            self.collection = None

    def setup_indexes(self):
        try:
            self.collection.create_index("username")
            self.collection.create_index("scan_date")
            self.collection.create_index([("username", 1), ("scan_date", -1)])
            self.collection.create_index("expires_at", expireAfterSeconds=0)
            print("✅ Scan indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create scan indexes: {e}")

    def save_scan_data(self, username, raw_text, cleaned_text, structured_nutrients,
                       image_filename=None, product_name=None, product_image_url=None):
        if self.collection is None:
            return None
        try:
            summary = self._create_nutrition_summary(structured_nutrients)
            scan_document = {
                "username": username,
                "scan_date": datetime.utcnow(),
                "product_info": {
                    "name": product_name or "Scanned Item",
                    "image_url": product_image_url
                },
                "scan_metadata": {
                    "image_filename": image_filename,
                    "nutrients_count": len(structured_nutrients) if structured_nutrients else 0,
                    "processing_timestamp": datetime.utcnow().isoformat()
                },
                "ocr_data": {
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text
                },
                "nutrition_analysis": {
                    "structured_nutrients": structured_nutrients,
                    "summary": summary
                },
                "system_info": {
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow() + timedelta(days=30),
                    "app_version": "1.0"
                }
            }
            result = self.collection.insert_one(scan_document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to save scan data: {e}")
            return None

    def _create_nutrition_summary(self, structured_nutrients):
        if not structured_nutrients:
            return {}
        summary = {
            "total_nutrients": len(structured_nutrients),
            "high_risk_count": len([n for n in structured_nutrients if n.get('status') == 'High']),
            "low_risk_count": len([n for n in structured_nutrients if n.get('status') == 'Low']),
            "normal_count": len([n for n in structured_nutrients if n.get('status') == 'Normal']),
            "unknown_count": len([n for n in structured_nutrients if n.get('status') == 'Unknown']),
            "categories": {}
        }
        for nutrient in structured_nutrients:
            category = nutrient.get('category', 'unknown')
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
        return summary

    def get_user_scans(self, username, limit=50):
        if self.collection is None:
            return []
        try:
            scans = list(self.collection.find(
                {"username": username},
                sort=[("scan_date", -1)]
            ).limit(limit))
            for scan in scans:
                scan['_id'] = str(scan['_id'])
                if isinstance(scan.get('scan_date'), datetime):
                    scan['scan_date'] = scan['scan_date'].isoformat()
            return scans
        except Exception as e:
            return []

    def get_scan_by_id(self, scan_id):
        if self.collection is None:
            return None
        try:
            scan = self.collection.find_one({"_id": ObjectId(scan_id)})
            if scan:
                scan['_id'] = str(scan['_id'])
                if isinstance(scan.get('scan_date'), datetime):
                    scan['scan_date'] = scan['scan_date'].isoformat()
            return scan
        except Exception as e:
            return None

    def delete_scan(self, scan_id, username):
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"_id": ObjectId(scan_id), "username": username})
            return result.deleted_count > 0
        except Exception as e:
            return False

    def get_user_scan_stats(self, username):
        if self.collection is None:
            return {"total_scans": 0, "recent_scans": 0, "latest_scan_date": None, "has_scans": False}
        try:
            total_scans = self.collection.count_documents({"username": username})
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_scans = self.collection.count_documents({
                "username": username,
                "scan_date": {"$gte": week_ago}
            })
            latest_scan = self.collection.find_one({"username": username}, sort=[("scan_date", -1)])
            latest_date = latest_scan['scan_date'].isoformat() if latest_scan and isinstance(latest_scan.get('scan_date'), datetime) else None
            return {
                "total_scans": total_scans,
                "recent_scans": recent_scans,
                "latest_scan_date": latest_date,
                "has_scans": total_scans > 0
            }
        except Exception as e:
            return {"total_scans": 0, "recent_scans": 0, "latest_scan_date": None, "has_scans": False}

    def cleanup_expired_scans(self):
        if self.collection is None:
            return 0
        try:
            result = self.collection.delete_many({"expires_at": {"$lt": datetime.utcnow()}})
            return result.deleted_count
        except Exception as e:
            return 0


# Initialize storage
credentials_storage = UserCredentialsStorage()
profile_storage = UserProfileStorage()
scan_storage = ScanStorage()


# ==================== UTILITY FUNCTIONS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def save_user_profile(data):
    return profile_storage.save_profile(data)

def load_user_health_profile(username):
    return profile_storage.get_profile(username)

def calculate_daily_needs(weight_kg, activity_level):
    if weight_kg is None:
        raise ValueError("Weight is required")
    weight_kg = float(weight_kg)
    if weight_kg <= 0:
        raise ValueError("Weight must be greater than 0")
    if not activity_level:
        raise ValueError("Activity level is required")
    multiplier = {'sedentary': 25, 'moderate': 30, 'active': 35}.get(activity_level, 30)
    calories = weight_kg * multiplier
    return {
        "calories": round(calories),
        "protein_g": round(weight_kg * 1.2),
        "fats_g": round((0.25 * calories) / 9),
        "carbs_g": round((0.50 * calories) / 4)
    }

def evaluate_nutrient_status_enhanced(nutrient_name, value):
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
        FROM ingredient_thresholds 
        WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
    """, (nutrient_name,))
    result = cursor.fetchone()
    if not result:
        name_mappings = {
            'sugar': ['sugar', 'added sugar', 'total sugar', 'total sugars'],
            'fat': ['fat', 'total fat', 'fats'],
            'saturated fat': ['saturated fat', 'sat fat'],
            'trans fat': ['trans fat'],
            'sodium': ['sodium', 'salt'],
            'carbohydrates': ['carbohydrates', 'carbs', 'total carbohydrates', 'total carbohydrate'],
            'protein': ['protein'],
            'fiber': ['fiber', 'dietary fiber'],
            'cholesterol': ['cholesterol'],
            'calcium': ['calcium'],
            'iron': ['iron'],
            'vitamin c': ['vitamin c', 'ascorbic acid'],
            'vitamin a': ['vitamin a'],
            'vitamin d': ['vitamin d'],
            'potassium': ['potassium']
        }
        for db_name, variations in name_mappings.items():
            if nutrient_name.lower() in [v.lower() for v in variations]:
                cursor.execute("""
                    SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
                    FROM ingredient_thresholds WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
                """, (db_name,))
                result = cursor.fetchone()
                if result:
                    break
    if not result:
        cursor.execute("""
            SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
            FROM ingredient_thresholds WHERE LOWER(name) LIKE LOWER(?)
        """, (f'%{nutrient_name}%',))
        result = cursor.fetchone()
    conn.close()
    if not result:
        return {"status": "Unknown", "message": f"No threshold data available for {nutrient_name}."}
    max_threshold, min_threshold, high_msg, low_msg = result
    if max_threshold is not None and value > max_threshold:
        return {"status": "High", "message": high_msg}
    elif min_threshold is not None and value < min_threshold:
        return {"status": "Low", "message": low_msg}
    else:
        return {"status": "Normal", "message": "This level is within the healthy range."}

def get_today_scan_visualization():
    username = session.get("username")
    if not username or scan_storage.collection is None:
        return []
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        today_scans = list(scan_storage.collection.find({
            "username": username,
            "scan_date": {"$gte": today_start, "$lt": today_end}
        }).sort("scan_date", -1))
        visualization_data = []
        for scan in today_scans:
            product_name = scan.get("product_info", {}).get("name", "Unknown Product")
            nutrients = scan.get("nutrition_analysis", {}).get("structured_nutrients", [])
            key_nutrients = {}
            for nutrient in nutrients:
                nutrient_name = nutrient.get("nutrient", "").lower()
                if nutrient_name in ["protein", "carbohydrates", "carbs", "fat", "fats", "sodium", "sugar"]:
                    key_nutrients[nutrient_name] = {
                        "value": nutrient.get("value", 0),
                        "unit": nutrient.get("unit", ""),
                        "status": nutrient.get("status", "Unknown")
                    }
            visualization_data.append({
                "product_name": product_name,
                "scan_time": scan.get("scan_date").isoformat() if isinstance(scan.get("scan_date"), datetime) else None,
                "nutrients": key_nutrients,
                "total_nutrients": len(nutrients)
            })
        return visualization_data
    except Exception as e:
        print(f"❌ Error fetching today's scans: {e}")
        return []

def get_comprehensive_ai_analysis():
    username = session.get("username")
    if not username:
        return "<p>No user logged in.</p>"
    if scan_storage.collection is None:
        return "<p>Database not connected.</p>"
    try:
        user_scans = list(scan_storage.collection.find({"username": username}))
        user_profile = load_user_health_profile(username)
        if not user_profile:
            return "<p>Please complete your profile for analysis.</p>"
    except Exception as e:
        return "<p>Could not fetch user data.</p>"
    if not user_scans:
        return "<p>No scans found. Please scan items to get an analysis.</p>"

    total_nutrients = {}
    risk_counts = {"High": 0, "Low": 0, "Normal": 0, "Unknown": 0}
    for scan in user_scans:
        for nutrient in scan.get("nutrition_analysis", {}).get("structured_nutrients", []):
            nutrient_name = nutrient.get("nutrient", "").lower()
            value = nutrient.get("value", 0)
            status = nutrient.get("status", "Unknown")
            if nutrient_name:
                if nutrient_name not in total_nutrients:
                    total_nutrients[nutrient_name] = {"total": 0, "count": 0, "unit": nutrient.get("unit", "")}
                total_nutrients[nutrient_name]["total"] += value
                total_nutrients[nutrient_name]["count"] += 1
            if status in risk_counts:
                risk_counts[status] += 1

    avg_nutrients = {
        n: {"average": round(d["total"] / d["count"], 2), "unit": d["unit"], "total": round(d["total"], 2)}
        for n, d in total_nutrients.items() if d["count"] > 0
    }
    analysis_data = {
        "user_profile": {
            "name": user_profile.get("full_name", "User"),
            "age": user_profile.get("age"),
            "gender": user_profile.get("gender"),
            "activity_level": user_profile.get("activity_level")
        },
        "risk_distribution": risk_counts,
        "average_nutrients": avg_nutrients
    }
    if user_profile.get("weight_kg") and user_profile.get("activity_level"):
        try:
            analysis_data["daily_recommendations"] = calculate_daily_needs(
                user_profile["weight_kg"], user_profile["activity_level"]
            )
        except Exception:
            pass

    data_json = json.dumps(analysis_data, indent=2)
    prompt = f"""
    You are an expert nutritionist providing a personalized health analysis. Analyze the following data:
    ```json
    {data_json}
    ```
    Generate your full response in **HTML format** with exactly these 5 sections:
    1. <h3>User Profile Overview</h3>
    2. <h3>Health Risk Distribution</h3>
    3. <h3>Nutrient & Daily Recommendation Insights</h3>
    4. <h3>Overall Health Summary</h3>
    5. <h3>Next Steps & Suggestions</h3>
    Each section MUST be wrapped in <div class="analysis-section">.
    Use <p>, <ul>/<li>, <strong>. DO NOT include <html>, <body>, or markdown fences.
    """
    try:
        if not client:
            return "<p>OpenAI client not configured.</p>"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a qualified nutritionist who writes professional, structured HTML reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.35,
            max_tokens=2000
        )
        raw_html = response.choices[0].message.content.strip()
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
        return raw_html.strip()
    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return "<p>Could not generate AI analysis. Please try again.</p>"

def calculate_health_score(aggregated_nutrients, daily_recommendations):
    score = 100
    nutrient_map = {
        'protein': ['protein'],
        'carbs': ['carbohydrates', 'carbs', 'total carbohydrates'],
        'fats': ['fat', 'fats', 'total fat'],
        'sodium': ['sodium'],
        'sugar': ['sugar', 'sugars', 'added sugar']
    }
    def get_nutrient_value(key):
        for alias in nutrient_map[key]:
            if alias in aggregated_nutrients:
                return aggregated_nutrients[alias]
        return 0
    for macro, goal in {'protein': daily_recommendations.get('protein_g', 1),
                        'carbs': daily_recommendations.get('carbs_g', 1),
                        'fats': daily_recommendations.get('fats_g', 1)}.items():
        actual = get_nutrient_value(macro)
        if goal > 0:
            score -= min(abs(actual - goal) / goal * 40, 20)
    sodium = get_nutrient_value('sodium')
    if sodium > 2300:
        score -= min((sodium - 2300) / 2300 * 40, 20)
    sugar = get_nutrient_value('sugar')
    if sugar > 50:
        score -= min((sugar - 50) / 50 * 40, 20)
    return max(0, round(score))

def get_historical_health_scores(username, period, recommendations):
    if scan_storage.collection is None:
        return {"labels": [], "scores": []}
    end_date = datetime.utcnow()
    if period == 'daily':
        start_date = end_date - timedelta(days=30)
        group_id = {"$dateToString": {"format": "%Y-%m-%d", "date": "$scan_date"}}
    elif period == 'weekly':
        start_date = end_date - timedelta(weeks=12)
        group_id = {"$dateToString": {"format": "%Y-%U", "date": "$scan_date"}}
    else:
        start_date = end_date - timedelta(days=365)
        group_id = {"$dateToString": {"format": "%Y-%m", "date": "$scan_date"}}

    pipeline = [
        {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
        {"$unwind": "$nutrition_analysis.structured_nutrients"},
        {"$group": {
            "_id": {"period": group_id, "nutrient": {"$toLower": "$nutrition_analysis.structured_nutrients.nutrient"}},
            "total_value": {"$sum": "$nutrition_analysis.structured_nutrients.value"}
        }},
        {"$group": {
            "_id": "$_id.period",
            "nutrients": {"$push": {"k": "$_id.nutrient", "v": "$total_value"}}
        }},
        {"$addFields": {"nutrients": {"$arrayToObject": "$nutrients"}}},
        {"$sort": {"_id": 1}}
    ]
    results = list(scan_storage.collection.aggregate(pipeline))
    scores, labels = [], []
    for result in results:
        nutrients = result['nutrients']
        if period == 'weekly':
            nutrients = {k: v / 7 for k, v in nutrients.items()}
        elif period == 'monthly':
            nutrients = {k: v / 30 for k, v in nutrients.items()}
        scores.append(calculate_health_score(nutrients, recommendations))
        labels.append(result['_id'])
    return {"labels": labels, "scores": scores}

def get_personalized_nutrient_analysis(nutrient_name, nutrient_value, nutrient_unit, nutrient_status, user_profile):
    if not client:
        return "AI analysis unavailable."
    prompt = f"""
You are a nutritionist providing personalized health advice.
USER PROFILE: Age: {user_profile.get('age','N/A')}, Gender: {user_profile.get('gender','N/A')},
Weight: {user_profile.get('weight_kg','N/A')} kg, Height: {user_profile.get('height_cm','N/A')} cm,
Activity: {user_profile.get('activity_level','N/A')}, Conditions: {user_profile.get('medical_conditions','None')},
Allergies: {user_profile.get('allergies','None')}
NUTRIENT: {nutrient_name}, Amount: {nutrient_value} {nutrient_unit}, Status: {nutrient_status}
Provide a brief 1-2 sentence personalized analysis. Return ONLY the analysis text.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise nutritionist providing personalized health insights in 1-2 sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unable to generate personalized analysis at this time."

def get_community_warnings(product_name):
    if not product_name or product_name == "Scanned from Label (OCR)":
        return {"total_reports": 0, "insights": [], "product_name": product_name, "warning_level": "none", "positive_reports": 0, "summary_message": "No community data available yet."}
    if scan_storage.collection is None:
        return {"total_reports": 0, "insights": [], "product_name": product_name, "warning_level": "none", "positive_reports": 0, "summary_message": "Unable to retrieve community data."}
    try:
        pipeline = [
            {"$match": {"product_info.name": product_name, "user_feedback.issue_report.description": {"$exists": True, "$ne": ""}}},
            {"$project": {"username": 1, "description": "$user_feedback.issue_report.description", "product_name": "$product_info.name"}}
        ]
        all_reports = list(scan_storage.collection.aggregate(pipeline))
        total_reports_count = len(all_reports)
        if total_reports_count == 0:
            return {"total_reports": 0, "insights": [], "product_name": product_name, "warning_level": "none", "positive_reports": 0, "summary_message": "No community reports found for this product."}

        topic_users = {}
        for report in all_reports:
            desc = report.get("description", "").strip()
            username = report.get("username")
            if not desc or not username:
                continue
            topic = analyze_report_text(desc)
            if topic not in topic_users:
                topic_users[topic] = set()
            topic_users[topic].add(username)

        topic_counts = {topic: len(users) for topic, users in topic_users.items()}
        NEGATIVE_TOPICS = {
            "Sickness / Nausea / Vomiting": {"desc": "nausea, vomiting, or stomach discomfort"},
            "Allergic Reaction / Rash / Itching": {"desc": "allergic reactions like rashes or itching"},
            "Bad Taste / Foul Smell": {"desc": "bad taste or foul smell"},
            "Packaging Defect / Foreign Object": {"desc": "packaging issues or foreign objects"},
            "Headache / Dizziness": {"desc": "headaches, dizziness, or fatigue"}
        }
        positive_count = topic_counts.get("Positive Feedback / No Issue", 0)
        negative_reports = {k: v for k, v in topic_counts.items() if k in NEGATIVE_TOPICS}
        total_negative_users = sum(negative_reports.values())
        all_reporting_users = set()
        for users in topic_users.values():
            all_reporting_users.update(users)
        total_unique_reporters = len(all_reporting_users)
        negative_user_pct = (total_negative_users / total_unique_reporters * 100) if total_unique_reporters else 0

        if total_negative_users >= 5 and negative_user_pct >= 70:
            warning_level = "high"
        elif total_negative_users >= 3 and negative_user_pct >= 40:
            warning_level = "medium"
        elif total_negative_users >= 1:
            warning_level = "low"
        else:
            warning_level = "none"

        insights = []
        for topic, count in sorted(negative_reports.items(), key=lambda x: x[1], reverse=True):
            topic_info = NEGATIVE_TOPICS[topic]
            if count >= 10:
                prefix, suffix, severity = "Many users have reported", "This issue seems widespread.", "widespread"
            elif count >= 5:
                prefix, suffix, severity = "Several users mentioned", "It could be a recurring issue.", "common"
            elif count >= 2:
                prefix, suffix, severity = "A few users noticed", "May not affect everyone.", "some"
            else:
                prefix, suffix, severity = "One user reported", "This may be an isolated incident.", "few"
            insights.append({"text": f"{prefix} {topic_info['desc']}. {suffix}", "severity": severity, "category": topic})

        if warning_level == "high":
            summary = f"HIGH RISK: {total_negative_users} of {total_unique_reporters} unique users reported major issues."
        elif warning_level == "medium":
            summary = "Moderate Risk: Some users reported recurring issues. Review before consumption."
        elif warning_level == "low":
            summary = "Low Risk: A few isolated reports exist, but overall community sentiment is neutral."
        elif positive_count > 0:
            summary = "Mostly Positive: Majority of users reported a good experience." if positive_count >= total_unique_reporters * 0.7 else "Mixed Reviews: Some users were satisfied, others had minor complaints."
        else:
            summary = "No significant feedback trends detected."

        return {
            "total_reports": total_reports_count,
            "total_unique_reporters": total_unique_reporters,
            "negative_reports": total_negative_users,
            "positive_reports": positive_count,
            "insights": insights,
            "product_name": product_name,
            "warning_level": warning_level,
            "summary_message": summary
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"total_reports": 0, "insights": [], "product_name": product_name, "warning_level": "none", "positive_reports": 0, "summary_message": "Error retrieving community warnings."}


# ==================== SERVE REACT APP (production only) ====================
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    """Serve React frontend - only used after npm run build"""
    # Don't intercept API, auth, or static file routes
    if path.startswith(('api/', 'auth/', 'static/')):
        return jsonify({"error": "Not found"}), 404
    dist_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(dist_path):
        return send_from_directory(app.static_folder, path)
    # Fall back to index.html for React Router
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "React build not found. Run: cd frontend && npm run build"}), 404


# ==================== AUTH ROUTES ====================
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            return redirect("http://localhost:5173/?error=google_failed")

        google_id = user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')
        username = email.split('@')[0] if email else f"google_user_{google_id}"

        existing_user = credentials_storage.get_user(username)
        if existing_user:
            if existing_user.get('auth_provider') != 'google':
                return redirect(f"http://localhost:5173/?error=username_taken")
            credentials_storage.collection.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            session.permanent = False
            session['username'] = username
            session['auth_provider'] = 'google'
            session['profile_picture'] = picture
            return redirect("http://localhost:5173/profile")
        else:
            credentials_storage.collection.insert_one({
                "username": username,
                "email": email,
                "google_id": google_id,
                "auth_provider": "google",
                "password": None,
                "profile_picture": picture,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            })
            profile_storage.save_profile({'username': username, 'full_name': name or '', 'email': email})
            session.permanent = False
            session['username'] = username
            session['auth_provider'] = 'google'
            session['profile_picture'] = picture
            return redirect("http://localhost:5173/profile-form")
    except Exception as e:
        print(f"❌ Google OAuth Error: {e}")
        return redirect(f"http://localhost:5173/?error=auth_failed")

limiter.exempt(google_callback)


@app.route('/auth/logout')
@login_required
def oauth_logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


# ==================== API ROUTES ====================

@app.route('/api/auth/status')
def auth_status():
    """Check if user is logged in — React calls this on load"""
    if 'username' in session:
        return jsonify({
            "authenticated": True,
            "username": session['username'],
            "auth_provider": session.get('auth_provider'),
            "profile_picture": session.get('profile_picture')
        })
    return jsonify({"authenticated": False})


@app.route('/api/login', methods=['POST'])
@limiter.limit("3 per minute")
def login():
    data = request.get_json() or request.form
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = credentials_storage.get_user(username)
    if not user:
        return jsonify({"error": "Account does not exist. Please sign up first."}), 404

    if user.get('auth_provider') == 'google':
        return jsonify({"error": "This account uses Google Sign-In. Please click 'Continue with Google'."}), 400

    hashed = user.get('password')
    if not hashed:
        return jsonify({"error": "Invalid account configuration."}), 500

    if check_password_hash(hashed, password):
        session.permanent = False
        session['username'] = username
        session['auth_provider'] = 'regular'
        return jsonify({"success": True, "message": "Logged in successfully!", "redirect": "/profile"})

    return jsonify({"error": "Incorrect password."}), 401


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or request.form
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if credentials_storage.user_exists(username):
        return jsonify({"error": "Username already exists."}), 409

    if profile_storage.save_profile({'username': username}) and credentials_storage.add_user(username, password):
        session.permanent = False
        session['username'] = username
        session['auth_provider'] = 'regular'
        return jsonify({"success": True, "message": "Account created successfully!", "redirect": "/profile-form"})

    if credentials_storage.add_user(username, password):
        session.permanent = False
        session['username'] = username
        return jsonify({"success": True, "message": "Account created successfully!", "redirect": "/profile-form"})

    return jsonify({"error": "Failed to create account. Please try again."}), 500


@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out."})


@app.route('/api/profile', methods=['GET'])
@login_required
def profile():
    username = session['username']
    profile_data = load_user_health_profile(username)
    if not profile_data:
        return jsonify({"error": "Profile not found", "redirect": "/profile-form"}), 404

    try:
        recommendation = calculate_daily_needs(profile_data['weight_kg'], profile_data['activity_level'])
    except (ValueError, KeyError, TypeError):
        return jsonify({"error": "Profile incomplete. Please update weight and activity level.", "redirect": "/edit-profile"}), 400

    today_scans_data = get_today_scan_visualization()
    today_totals = {"protein": 0, "carbs": 0, "fats": 0}
    for scan in today_scans_data:
        for nutrient_name, nutrient_data in scan["nutrients"].items():
            if nutrient_name == "protein":
                today_totals["protein"] += nutrient_data["value"]
            elif nutrient_name in ["carbohydrates", "carbs"]:
                today_totals["carbs"] += nutrient_data["value"]
            elif nutrient_name in ["fat", "fats"]:
                today_totals["fats"] += nutrient_data["value"]

    percent = lambda val, ref: round((val / ref) * 100) if ref else 0
    scan_stats = scan_storage.get_user_scan_stats(username)

    return jsonify({
        "profile": profile_data,
        "intake": {
            "protein_g": round(today_totals['protein'], 1),
            "carbs_g": round(today_totals['carbs'], 1),
            "fats_g": round(today_totals['fats'], 1)
        },
        "recommendation": recommendation,
        "percentages": {
            "protein": percent(today_totals['protein'], recommendation['protein_g']),
            "carbs": percent(today_totals['carbs'], recommendation['carbs_g']),
            "fats": percent(today_totals['fats'], recommendation['fats_g'])
        },
        "today_scans": today_scans_data,
        "scan_stats": scan_stats,
        "profile_picture": session.get('profile_picture')
    })


@app.route('/api/profile', methods=['POST'])
@login_required
def save_profile():
    data = request.get_json() or request.form
    profile_data = {
        'username': session['username'],
        'full_name': data.get('full_name', ''),
        'height_cm': data.get('height_cm', ''),
        'weight_kg': data.get('weight_kg', ''),
        'age': data.get('age', ''),
        'gender': data.get('gender', ''),
        'activity_level': data.get('activity_level', ''),
        'medical_conditions': data.get('medical_conditions', ''),
        'allergies': data.get('allergies', ''),
    }
    if save_user_profile(profile_data):
        return jsonify({"success": True, "message": "Profile saved successfully!", "redirect": "/profile"})
    return jsonify({"error": "Failed to save profile. Please try again."}), 500


@app.route('/api/dashboard', methods=['GET'])
@login_required
def dashboard():
    username = session['username']
    scan_stats = scan_storage.get_user_scan_stats(username)
    return jsonify({"username": username, "scan_stats": scan_stats})


@app.route('/api/scan-label', methods=['POST'])
@login_required
def scan_label():
    if 'label_image' in request.files and request.files['label_image'].filename != '':
        file = request.files['label_image']
        scan_type = "label"
    elif 'barcode_image' in request.files and request.files['barcode_image'].filename != '':
        file = request.files['barcode_image']
        scan_type = "barcode"
    else:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files allowed"}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{timestamp}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename).replace("\\", "/")

    try:
        image_stream = BytesIO(file.read())
        img = Image.open(image_stream)
        if img.format not in ("PNG", "JPEG"):
            return jsonify({"error": f"Unsupported format: {img.format}. Please upload JPG or PNG."}), 400
        img.verify()
        image_stream.seek(0)
        img = Image.open(image_stream).convert("RGB")
        img.save(filepath)

        username = session.get('username')

        if scan_type == "label":
            cleaned_text, structured_nutrients = process_label_image(filepath)
            user_profile = load_user_health_profile(username)

            for nutrient in structured_nutrients:
                try:
                    value = float(re.sub(r'[^\d.]', '', str(nutrient['value'])))
                except (ValueError, TypeError):
                    value = 0.0
                nutrient_name = nutrient.get('nutrient', '').strip()
                status_info = evaluate_nutrient_status_enhanced(nutrient_name, value)
                nutrient['value'] = value
                nutrient['status'] = status_info['status']
                nutrient['message'] = status_info['message']
                if user_profile:
                    nutrient['ai_analysis'] = get_personalized_nutrient_analysis(
                        nutrient_name, value, nutrient.get('unit', 'g'), status_info['status'], user_profile
                    )
                else:
                    nutrient['ai_analysis'] = "Complete your profile to get personalized insights."

            scan_id = scan_storage.save_scan_data(
                username=username, raw_text=cleaned_text, cleaned_text=cleaned_text,
                structured_nutrients=structured_nutrients, image_filename=unique_filename,
                product_name="Scanned from Label (OCR)", product_image_url=None
            )

            return jsonify({
                "success": True,
                "scan_type": "label",
                "scan_id": scan_id,
                "raw_text": cleaned_text,
                "structured_nutrients": structured_nutrients
            })

        elif scan_type == "barcode":
            processed_data = process_nutrition_label(filepath)
            product_name = processed_data.get('product_name')
            community_warnings = get_community_warnings(product_name) if product_name else {}
            return jsonify({
                "success": True,
                "scan_type": "barcode",
                "scan_data": processed_data,
                "community_warnings": community_warnings
            })

    except Exception as e:
        print(f"❌ Error handling image: {e}")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 500


@app.route('/api/add-to-diet', methods=['POST'])
@login_required
def add_to_diet():
    username = session.get('username')
    data = request.get_json()
    product_name = data.get('product_name')
    product_image_url = data.get('product_image_url')
    structured_nutrients = data.get('structured_nutrients', [])
    image_filename = data.get('image_filename')

    scan_id = scan_storage.save_scan_data(
        username=username,
        raw_text=f"Barcode scan for: {product_name}",
        cleaned_text=f"Product: {product_name}",
        structured_nutrients=structured_nutrients,
        image_filename=image_filename,
        product_name=product_name,
        product_image_url=product_image_url
    )
    if scan_id:
        return jsonify({"status": "success", "message": "Item added to your diet successfully!"})
    return jsonify({"status": "warning", "message": "Could not add item to your diet."}), 500


@app.route('/api/scans', methods=['GET'])
@login_required
def get_all_scans():
    username = session['username']
    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503
    try:
        scans = list(scan_storage.collection.find({"username": username}).sort("scan_date", -1))
        for scan in scans:
            scan['_id'] = str(scan['_id'])
            if isinstance(scan.get('scan_date'), datetime):
                scan['scan_date'] = scan['scan_date'].isoformat()

        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_scans = len([s for s in scans if 'scan_date' in s and datetime.fromisoformat(s['scan_date']) >= week_ago])
        high_risk_count = sum(s.get('nutrition_analysis', {}).get('summary', {}).get('high_risk_count', 0) for s in scans)

        return jsonify({
            "success": True,
            "scans": scans,
            "stats": {"total_scans": len(scans), "recent_scans": recent_scans, "high_risk_count": high_risk_count}
        })
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to fetch scans"}), 500


@app.route('/api/scans/<scan_id>', methods=['GET'])
@login_required
def get_scan(scan_id):
    scan = scan_storage.get_scan_by_id(scan_id)
    if not scan or scan['username'] != session['username']:
        return jsonify({"error": "Scan not found or access denied."}), 404
    return jsonify({"success": True, "scan": scan})


@app.route('/api/scans/<scan_id>', methods=['DELETE'])
@login_required
def delete_scan(scan_id):
    username = session['username']
    try:
        obj_id = ObjectId(scan_id)
    except (InvalidId, TypeError):
        return jsonify({"success": False, "error": "Invalid scan ID format"}), 400

    result = scan_storage.collection.delete_one({"_id": obj_id, "username": username})
    if result.deleted_count > 0:
        return jsonify({"success": True, "message": "Scan deleted successfully"})
    return jsonify({"success": False, "error": "Scan not found or access denied"}), 404


@app.route('/api/scans/<scan_id>/report', methods=['POST'])
@login_required
def submit_report(scan_id):
    username = session['username']
    if scan_storage.collection is None:
        return jsonify({"success": False, "error": "Database not available"}), 503
    try:
        obj_id = ObjectId(scan_id)
    except (InvalidId, TypeError):
        return jsonify({"success": False, "error": "Invalid scan ID format"}), 400

    report_data = request.get_json()
    if not report_data or not report_data.get('description', '').strip():
        return jsonify({"success": False, "error": "Report description is required"}), 400

    report_payload = {
        "reported_at": datetime.utcnow(),
        "description": report_data['description'].strip(),
        "severity": report_data.get('severity'),
        "contact_consent": report_data.get('contact_consent', False),
        "status": "submitted"
    }
    result = scan_storage.collection.update_one(
        {"_id": obj_id, "username": username},
        {"$set": {"user_feedback.issue_report": report_payload}}
    )
    if result.matched_count > 0:
        return jsonify({"success": True, "message": "Report submitted successfully"})
    return jsonify({"success": False, "error": "Scan not found or access denied"}), 404


@app.route('/api/ai-analysis', methods=['POST'])
@login_required
def get_ai_analysis():
    try:
        analysis_html = get_comprehensive_ai_analysis()
        return jsonify({"success": True, "analysis": analysis_html})
    except Exception as e:
        return jsonify({"error": "Failed to generate analysis"}), 500


@app.route('/api/health-score/<period>', methods=['GET'])
@login_required
def get_health_score_data(period='weekly'):
    username = session['username']
    profile = load_user_health_profile(username)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    if not all(k in profile and profile[k] for k in ['weight_kg', 'activity_level']):
        return jsonify({"error": "Profile incomplete"}), 400
    if period not in ['daily', 'weekly', 'monthly']:
        return jsonify({"error": "Invalid period"}), 400
    try:
        recommendations = calculate_daily_needs(profile['weight_kg'], profile['activity_level'])
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    data = get_historical_health_scores(username, period, recommendations)
    return jsonify({
        "labels": data["labels"],
        "actual_scores": data["scores"],
        "goal_scores": [85] * len(data["labels"])
    })


@app.route('/api/scan-count/<period>', methods=['GET'])
@login_required
def get_scan_count_data(period='weekly'):
    username = session['username']
    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503
    if period not in ['daily', 'weekly', 'monthly']:
        return jsonify({"error": "Invalid period"}), 400

    end_date = datetime.utcnow()
    if period == 'daily':
        start_date = end_date - timedelta(days=30)
        group_id = {"$dateToString": {"format": "%Y-%m-%d", "date": "$scan_date"}}
    elif period == 'weekly':
        start_date = end_date - timedelta(weeks=12)
        group_id = {"$dateToString": {"format": "%Y-W%U", "date": "$scan_date"}}
    else:
        start_date = end_date - timedelta(days=365)
        group_id = {"$dateToString": {"format": "%Y-%m", "date": "$scan_date"}}

    try:
        pipeline = [
            {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
            {"$group": {"_id": group_id, "scan_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        results = list(scan_storage.collection.aggregate(pipeline))
        labels = []
        for r in results:
            label = r['_id']
            if period == 'weekly':
                try:
                    year, week = label.split('-W')
                    label = f"Week {week}, {year}"
                except:
                    pass
            elif period == 'monthly':
                try:
                    year, month = label.split('-')
                    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    label = f"{months[int(month)-1]} {year}"
                except:
                    pass
            labels.append(label)
        scan_counts = [r['scan_count'] for r in results]
        return jsonify({"labels": labels, "scan_counts": scan_counts, "total_scans": sum(scan_counts)})
    except Exception as e:
        return jsonify({"error": "Failed to fetch scan data"}), 500


@app.route('/api/delete-account', methods=['DELETE'])
@login_required
def delete_account():
    username = session['username']
    try:
        credentials_storage.collection.delete_one({"username": username})
        profile_storage.delete_profile(username)
        scan_storage.collection.delete_many({"username": username})
        session.clear()
        return jsonify({"success": True, "message": "Account deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/community-warnings/<product_name>', methods=['GET'])
@login_required
def community_warnings(product_name):
    warnings = get_community_warnings(product_name)
    return jsonify(warnings)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many attempts. Please try again later."}), 429


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True, use_reloader=False)