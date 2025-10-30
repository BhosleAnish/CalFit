from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import  os, re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ocr_utils import process_label_image,extract_nutrients
from PIL import Image
import pillow_avif  # enables AVIF support in Pillow
from io import BytesIO
import sqlite3
import pymongo
from bson import ObjectId
from bson.errors import InvalidId
from process_label import process_nutrition_label
from datetime import datetime, timedelta
import openai
import json
from flask_cors import CORS
from flask import jsonify, request # Make sure jsonify and request are imported
from nlp_analyzer import analyze_report_text # <-- ADD THIS
load_dotenv()


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

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
CORS(app, 
     origins=["http://localhost:5173"], 
     supports_credentials=True # This is crucial for session cookies
)
csrf = CSRFProtect(app)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="redis://localhost:6379"
)

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

# File paths
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
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
            
            # Test connection
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

# ==================== USER CREDENTIALS STORAGE CLASS ====================
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
        """Create indexes for better performance"""
        try:
            self.collection.create_index("username", unique=True)
            print("✅ User credentials indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create credentials indexes: {e}")
    
    def add_user(self, username, password):
        """Add new user credentials to MongoDB"""
        if self.collection is None:
            print("❌ MongoDB not available, cannot add user")
            return False
        
        try:
            user_document = {
                "username": username,
                "password": generate_password_hash(password),
                "created_at": datetime.utcnow()
            }
            
            self.collection.insert_one(user_document)
            print(f"✅ User created: {username}")
            return True
            
        except pymongo.errors.DuplicateKeyError:
            print(f"⚠️ User already exists: {username}")
            return False
        except Exception as e:
            print(f"❌ Failed to add user: {e}")
            return False
    
    def get_user(self, username):
        """Get user credentials from MongoDB"""
        if self.collection is None:
            return None
        
        try:
            user = self.collection.find_one({"username": username})
            return user
        except Exception as e:
            print(f"❌ Failed to retrieve user: {e}")
            return None
    
    def user_exists(self, username):
        """Check if user exists"""
        if self.collection is None:
            return False
        
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            print(f"❌ Failed to check user existence: {e}")
            return False
    
    def get_all_users(self):
        """Get all usernames (admin function)"""
        if self.collection is None:
            return {}
        
        try:
            users = {}
            for user in self.collection.find({}, {'username': 1, 'password': 1, '_id': 0}):
                users[user['username']] = user['password']
            return users
        except Exception as e:
            print(f"❌ Failed to retrieve all users: {e}")
            return {}

# ==================== USER PROFILE STORAGE CLASS ====================
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
        """Create indexes for better performance"""
        try:
            self.collection.create_index("username", unique=True)
            self.collection.create_index("created_at")
            print("✅ User profile indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create profile indexes: {e}")
    
    def save_profile(self, profile_data):
        """Save or update user profile in MongoDB"""
        if self.collection is None:
            print("❌ MongoDB not available, cannot save profile")
            return False
        
        try:
            username = profile_data.get('username')
            if not username:
                print("❌ Username is required to save profile")
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
            
            existing_profile = self.collection.find_one({"username": username})
            
            if existing_profile:
                profile_document['created_at'] = existing_profile.get('created_at', datetime.utcnow())
                result = self.collection.update_one(
                    {"username": username},
                    {"$set": profile_document}
                )
                print(f"✅ Profile updated for user: {username}")
                return result.modified_count > 0 or result.matched_count > 0
            else:
                profile_document['created_at'] = datetime.utcnow()
                result = self.collection.insert_one(profile_document)
                print(f"✅ Profile created for user: {username}")
                return result.inserted_id is not None
                
        except Exception as e:
            print(f"❌ Failed to save profile: {e}")
            return False
    
    def get_profile(self, username):
        """Retrieve user profile from MongoDB"""
        if self.collection is None:
            return None
        
        try:
            profile = self.collection.find_one({"username": username})
            if profile:
                profile.pop('_id', None)
                return profile
            return None
        except Exception as e:
            print(f"❌ Failed to retrieve profile: {e}")
            return None
    
    def delete_profile(self, username):
        """Delete user profile"""
        if self.collection is None:
            return False
        
        try:
            result = self.collection.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Failed to delete profile: {e}")
            return False
    
    def profile_exists(self, username):
        """Check if profile exists"""
        if self.collection is None:
            return False
        
        try:
            return self.collection.count_documents({"username": username}) > 0
        except Exception as e:
            print(f"❌ Failed to check profile existence: {e}")
            return False

# ==================== SCAN STORAGE CLASS ====================
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
        """Create indexes for better performance and automatic cleanup"""
        try:
            self.collection.create_index("username")
            self.collection.create_index("scan_date")
            self.collection.create_index([("username", 1), ("scan_date", -1)])
            self.collection.create_index("expires_at", expireAfterSeconds=0)
            print("✅ Scan indexes created successfully")
        except Exception as e:
            print(f"⚠️ Could not create scan indexes: {e}")
    
    def save_scan_data(self, username, raw_text, cleaned_text, structured_nutrients, image_filename=None, product_name=None, product_image_url=None):
        """Save scan data to MongoDB"""
        if self.collection is None:
            print("❌ MongoDB not available, cannot save scan data")
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
            print(f"✅ Scan data saved to MongoDB with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save scan data to MongoDB: {e}")
            return None
    
    def _create_nutrition_summary(self, structured_nutrients):
        """Create a summary of nutrition data"""
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
            if category not in summary["categories"]:
                summary["categories"][category] = 0
            summary["categories"][category] += 1
        
        return summary
    
    def get_user_scans(self, username, limit=50):
        """Retrieve all scans for a specific user"""
        if self.collection is None:
            return []
        
        try:
            scans = list(self.collection.find(
                {"username": username},
                sort=[("scan_date", -1)]
            ).limit(limit))
            
            for scan in scans:
                scan['_id'] = str(scan['_id'])
            
            return scans
            
        except Exception as e:
            print(f"❌ Failed to retrieve user scans: {e}")
            return []
    
    def get_scan_by_id(self, scan_id):
        """Get a specific scan by its MongoDB ID"""
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
    
    def delete_scan(self, scan_id, username):
        """Delete a specific scan"""
        if self.collection is None:
            return False
        
        try:
            result = self.collection.delete_one({
                "_id": ObjectId(scan_id),
                "username": username
            })
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"❌ Failed to delete scan: {e}")
            return False
    
    def get_user_scan_stats(self, username):
        """Get statistics about user's scans"""
        if self.collection is None:
            return {
                "total_scans": 0,
                "recent_scans": 0,
                "latest_scan_date": None,
                "has_scans": False
            }
        
        try:
            total_scans = self.collection.count_documents({"username": username})
            
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_scans = self.collection.count_documents({
                "username": username,
                "scan_date": {"$gte": week_ago}
            })
            
            latest_scan = self.collection.find_one(
                {"username": username},
                sort=[("scan_date", -1)]
            )
            
            latest_date = latest_scan['scan_date'] if latest_scan else None
            
            return {
                "total_scans": total_scans,
                "recent_scans": recent_scans,
                "latest_scan_date": latest_date,
                "has_scans": total_scans > 0
            }
            
        except Exception as e:
            print(f"❌ Failed to get user scan stats: {e}")
            return {
                "total_scans": 0,
                "recent_scans": 0,
                "latest_scan_date": None,
                "has_scans": False
            }
    
    def cleanup_expired_scans(self):
        """Remove expired scans"""
        if self.collection is None:
            return 0
        
        try:
            result = self.collection.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })
            
            print(f"🧹 Cleaned up {result.deleted_count} expired scans")
            return result.deleted_count
            
        except Exception as e:
            print(f"❌ Failed to cleanup expired scans: {e}")
            return 0

# Initialize storage systems
credentials_storage = UserCredentialsStorage()
profile_storage = UserProfileStorage()
scan_storage = ScanStorage()

# ==================== UTILITY FUNCTIONS ====================

def load_users():
    """Load users from MongoDB"""
    return credentials_storage.get_all_users()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def add_user(username, password):
    """Add user to MongoDB"""
    return credentials_storage.add_user(username, password)

def save_user_profile(data):
    """Save user profile to MongoDB"""
    return profile_storage.save_profile(data)

def load_user_health_profile(username):
    """Load user profile from MongoDB"""
    return profile_storage.get_profile(username)

def calculate_daily_needs(weight_kg, activity_level):
    weight_kg = float(weight_kg)
    multiplier = {'sedentary': 25, 'moderate': 30, 'active': 35}.get(activity_level, 30)
    calories = weight_kg * multiplier
    return {
        "calories": round(calories),
        "protein_g": round(weight_kg * 1.2),
        "fats_g": round((0.25 * calories) / 9),
        "carbs_g": round((0.50 * calories) / 4)
    }

def evaluate_nutrient_status_enhanced(nutrient_name, value):
    """Enhanced function to evaluate nutrient status with better name matching"""
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
                    FROM ingredient_thresholds 
                    WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
                """, (db_name,))
                result = cursor.fetchone()
                if result:
                    break
    
    if not result:
        cursor.execute("""
            SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
            FROM ingredient_thresholds 
            WHERE LOWER(name) LIKE LOWER(?)
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
    """Get today's scans for visualization on profile page"""
    username = session.get("username")
    if not username or scan_storage.collection is None:
        return []

    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        today_scans = list(scan_storage.collection.find({
            "username": username,
            "scan_date": {
                "$gte": today_start,
                "$lt": today_end
            }
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
                "scan_time": scan.get("scan_date"),
                "nutrients": key_nutrients,
                "total_nutrients": len(nutrients)
            })

        return visualization_data

    except Exception as e:
        print(f"❌ Error fetching today's scans: {e}")
        return []

def get_comprehensive_ai_analysis():
    """Generates a personalized, doctor-style analysis (refined version)"""
    username = session.get("username")
    if not username:
        return "<h2>Error</h2><p>No user logged in.</p>"

    if scan_storage.collection is None:
        return "<h2>AI Analysis Unavailable</h2><p>Database not connected.</p>"

    try:
        user_scans = list(scan_storage.collection.find({"username": username}))
        user_profile = load_user_health_profile(username)
        
        if not user_profile:
            return "<h2>Profile Required</h2><p>Please complete your profile for analysis.</p>"
            
    except Exception as e:
        print(f"❌ Could not fetch data: {e}")
        return "<h2>Error</h2><p>Could not fetch user data.</p>"

    if not user_scans:
        return "<h2>No Data</h2><p>No scans found. Please scan items to get an analysis.</p>"

    total_nutrients = {}
    risk_counts = {"High": 0, "Low": 0, "Normal": 0, "Unknown": 0}
    scan_count = len(user_scans)
    
    for scan in user_scans:
        structured_nutrients = scan.get("nutrition_analysis", {}).get("structured_nutrients", [])
        for nutrient in structured_nutrients:
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
    
    avg_nutrients = {}
    for nutrient, data in total_nutrients.items():
        if data["count"] > 0:
            avg_nutrients[nutrient] = {
                "average": round(data["total"] / data["count"], 2),
                "unit": data["unit"],
                "total": round(data["total"], 2)
            }

    analysis_data = {
        "user_profile": {
            "name": user_profile.get("full_name", "User"),
            "age": user_profile.get("age"),
            "gender": user_profile.get("gender"),
            "activity_level": user_profile.get("activity_level")
        },
        "risk_distribution": risk_counts,
        "average_nutrients": avg_nutrients  # kept for AI to interpret, not shown directly
    }

    if user_profile.get("weight_kg") and user_profile.get("activity_level"):
        try:
            analysis_data["daily_recommendations"] = calculate_daily_needs(
                user_profile["weight_kg"],
                user_profile["activity_level"]
            )
        except Exception as e:
            print(f"Warning: Could not calculate daily needs: {e}")

    data_json = json.dumps(analysis_data, indent=2)

    # ✅ New Prompt with merged and simplified sections
    prompt = f"""
    You are an expert nutritionist providing a personalized health analysis. Analyze the following data:

    ```json
    {data_json}
    ```

    Generate your full response in **HTML format** with exactly these 5 sections:
    1. <h3>User Profile Overview</h3> — summarize user's basic info and context.
    2. <h3>Health Risk Distribution</h3> — explain what the High, Normal, Low, and Unknown counts mean. 
       For each category, give one explanatory sentence (e.g., "Normal risk (84 cases) means that most nutrients are within ideal levels.").
    3. <h3>Nutrient & Daily Recommendation Insights</h3> — merge both analyses here.
       Present key nutrients as bullet points, each being a **1-sentence observation** comparing user's average intake vs daily needs, written like a doctor's remark.
       ⚠️ For every nutrient mentioned, explicitly include:
       - The **average intake** (e.g., "average protein intake is 45g")
       - The **recommended intake** (e.g., "recommended is 60g")
       - And a **brief interpretation** ("which indicates a mild deficiency")
       Example format:
       <li>Average <strong>Calories</strong> intake is 478 kcal, compared to the recommended 1300 kcal — indicating a lower energy intake than ideal.</li>
    4. <h3>Overall Health Summary</h3> — give a short summary (2–3 sentences) of the user's overall diet quality.
    5. <h3>Next Steps & Suggestions</h3> — provide 3–4 bullet points with actionable guidance.

    Formatting rules:
    - Each section MUST be wrapped in <div class="analysis-section">.
    - Use <p> for normal text, <ul>/<li> for bullet points, and <strong> for key terms.
    - DO NOT include <html>, <body>, or markdown fences.
    """


    try:
        if not client:
            return "<h2>AI Analysis Unavailable</h2><p>OpenAI client not configured.</p>"
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a qualified nutritionist who writes professional, structured HTML reports. "
                        "Provide context, avoid numbers alone, and make insights sound like doctor explanations."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.35,
            max_tokens=2000
        )
        
        raw_html = response.choices[0].message.content

        cleaned_html = raw_html.strip()
        if cleaned_html.startswith("```html"):
            cleaned_html = cleaned_html[7:]
        if cleaned_html.endswith("```"):
            cleaned_html = cleaned_html[:-3]
        
        return cleaned_html.strip()

    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return "<h2>Analysis Error</h2><p>Could not generate AI analysis. Please try again.</p>"


def calculate_health_score(aggregated_nutrients, daily_recommendations):
    """Calculates a health score from 0-100"""
    score = 100
    
    nutrient_map = {
        'protein': ['protein'],
        'carbs': ['carbohydrates', 'carbs', 'total carbohydrates'],
        'fats': ['fat', 'fats', 'total fat'],
        'sodium': ['sodium'],
        'sugar': ['sugar', 'sugars', 'added sugar']
    }

    def get_nutrient_value(nutrient_key):
        for alias in nutrient_map[nutrient_key]:
            if alias in aggregated_nutrients:
                return aggregated_nutrients[alias]
        return 0

    macro_goals = {
        'protein': daily_recommendations.get('protein_g', 1),
        'carbs': daily_recommendations.get('carbs_g', 1),
        'fats': daily_recommendations.get('fats_g', 1)
    }

    for macro, goal in macro_goals.items():
        actual = get_nutrient_value(macro)
        if goal > 0:
            deviation = abs(actual - goal) / goal
            penalty = min(deviation * 40, 20)
            score -= penalty

    sodium_actual = get_nutrient_value('sodium')
    if sodium_actual > 2300:
        excess_ratio = (sodium_actual - 2300) / 2300
        penalty = min(excess_ratio * 40, 20)
        score -= penalty
        
    sugar_actual = get_nutrient_value('sugar')
    if sugar_actual > 50:
        excess_ratio = (sugar_actual - 50) / 50
        penalty = min(excess_ratio * 40, 20)
        score -= penalty
        
    return max(0, round(score))

def get_historical_health_scores(username, period, recommendations):
    """Aggregates scan data over a period and calculates health scores"""
    if scan_storage.collection is None:
        return {"labels": [], "scores": []}

    end_date = datetime.utcnow()
    if period == 'daily':
        start_date = end_date - timedelta(days=30)
        date_format = "%Y-%m-%d"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
    elif period == 'weekly':
        start_date = end_date - timedelta(weeks=12)
        date_format = "%Y-%U"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
    else:
        start_date = end_date - timedelta(days=365)
        date_format = "%Y-%m"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
    
    pipeline = [
        {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
        {"$unwind": "$nutrition_analysis.structured_nutrients"},
        {"$group": {
            "_id": {
                "period": group_id,
                "nutrient": {"$toLower": "$nutrition_analysis.structured_nutrients.nutrient"}
            },
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
    
    scores = []
    labels = []
    for result in results:
        period_label = result['_id']
        aggregated_nutrients = result['nutrients']
        
        if period == 'weekly':
            for k, v in aggregated_nutrients.items():
                aggregated_nutrients[k] = v / 7
        elif period == 'monthly':
            for k, v in aggregated_nutrients.items():
                aggregated_nutrients[k] = v / 30

        score = calculate_health_score(aggregated_nutrients, recommendations)
        scores.append(score)
        labels.append(period_label)

    return {"labels": labels, "scores": scores}

def get_personalized_nutrient_analysis(nutrient_name, nutrient_value, nutrient_unit, nutrient_status, user_profile):
    """Generate personalized AI analysis for a specific nutrient"""
    if not client:
        return "AI analysis unavailable. Please configure OpenAI API key."
    
    user_context = {
        "age": user_profile.get('age', 'N/A'),
        "gender": user_profile.get('gender', 'N/A'),
        "weight_kg": user_profile.get('weight_kg', 'N/A'),
        "height_cm": user_profile.get('height_cm', 'N/A'),
        "activity_level": user_profile.get('activity_level', 'N/A'),
        "medical_conditions": user_profile.get('medical_conditions', 'None'),
        "allergies": user_profile.get('allergies', 'None')
    }
    
    prompt = f"""
You are a nutritionist providing personalized health advice.

USER PROFILE:
- Age: {user_context['age']}
- Gender: {user_context['gender']}
- Weight: {user_context['weight_kg']} kg
- Height: {user_context['height_cm']} cm
- Activity Level: {user_context['activity_level']}
- Medical Conditions: {user_context['medical_conditions']}
- Allergies: {user_context['allergies']}

NUTRIENT INFORMATION:
- Nutrient: {nutrient_name}
- Amount: {nutrient_value} {nutrient_unit}
- Status: {nutrient_status}

Provide a brief, personalized 1-2 sentence analysis of how this nutrient level affects THIS SPECIFIC USER, considering their profile. Be direct and actionable.

Format: Return ONLY the analysis text, no headers or extra formatting.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise nutritionist providing personalized health insights in 1-2 sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ AI analysis failed for {nutrient_name}: {e}")
        return "Unable to generate personalized analysis at this time."
# REPLACE your old 'get_community_warnings' function with this
# In app.py, replace your get_community_warnings with this:

def get_community_warnings(product_name):
    """
    Analyzes all user feedback for a specific product.
    Returns aggregated warnings, insights, and overall sentiment.
    
    ROBUSTNESS: This function counts *unique users* per topic to
    prevent a single user from skewing the results.
    """

    print(f"DEBUG: Looking for reports for product: '{product_name}'")

    # 1. Basic validation
    if not product_name or product_name == "Scanned from Label (OCR)":
        return {
            "total_reports": 0,
            "insights": [],
            "product_name": product_name,
            "warning_level": "none",
            "positive_reports": 0,
            "summary_message": "No community data available yet."
        }

    if scan_storage.collection is None:
        print("ERROR: No database connection.")
        return {
            "total_reports": 0,
            "insights": [],
            "product_name": product_name,
            "warning_level": "none",
            "positive_reports": 0,
            "summary_message": "Unable to retrieve community data."
        }

    try:
        # 2. Fetch reports (with username)
        pipeline = [
            {"$match": {
                "product_info.name": product_name,
                "user_feedback.issue_report.description": {"$exists": True, "$ne": ""}
            }},
            {"$project": {
                "username": 1,
                "description": "$user_feedback.issue_report.description",
                "product_name": "$product_info.name"
            }}
        ]

        all_reports = list(scan_storage.collection.aggregate(pipeline))
        total_reports_count = len(all_reports)
        print(f"DEBUG: Found {total_reports_count} total reports for '{product_name}'")

        if total_reports_count == 0:
            return {
                "total_reports": 0,
                "insights": [],
                "product_name": product_name,
                "warning_level": "none",
                "positive_reports": 0,
                "summary_message": "No community reports found for this product."
            }

        # 3. Categorize reports with NLP (by unique user)
        topic_users = {}  # e.g., {"Sickness...": {"user_A", "user_B"}}
        
        for report in all_reports:
            desc = report.get("description", "").strip()
            username = report.get("username")
            
            if not desc or not username:
                continue
                
            topic = analyze_report_text(desc)
            
            # Add the user to a set for that topic
            if topic not in topic_users:
                topic_users[topic] = set()
            topic_users[topic].add(username)  # A set automatically handles duplicates

        # Now, create counts based on the *number of unique users*
        topic_counts = {topic: len(users) for topic, users in topic_users.items()}

        print(f"DEBUG: Unique user topic counts: {topic_counts}")

        # 4. Define known issue types (removed emojis - icons handled by CSS)
        NEGATIVE_TOPICS = {
            "Sickness / Nausea / Vomiting": {"desc": "nausea, vomiting, or stomach discomfort"},
            "Allergic Reaction / Rash / Itching": {"desc": "allergic reactions like rashes or itching"},
            "Bad Taste / Foul Smell": {"desc": "bad taste or foul smell"},
            "Packaging Defect / Foreign Object": {"desc": "packaging issues or foreign objects"},
            "Headache / Dizziness": {"desc": "headaches, dizziness, or fatigue"}
        }

        positive_count = topic_counts.get("Positive Feedback / No Issue", 0)

        # 5. Collect negative reports
        negative_reports = {k: v for k, v in topic_counts.items() if k in NEGATIVE_TOPICS}
        total_negative_users = sum(negative_reports.values())
        
        # Get total unique users who submitted any report
        all_reporting_users = set()
        for users in topic_users.values():
            all_reporting_users.update(users)
        total_unique_reporters = len(all_reporting_users)

        negative_user_pct = (total_negative_users / total_unique_reporters * 100) if total_unique_reporters else 0

        # 6. Determine warning level (based on unique users)
        if total_negative_users >= 5 and negative_user_pct >= 70:
            warning_level = "high"
        elif total_negative_users >= 3 and negative_user_pct >= 40:
            warning_level = "medium"
        elif total_negative_users >= 1:
            warning_level = "low"
        else:
            warning_level = "none"

        print(f"WARNING LEVEL: {warning_level} ({total_negative_users}/{total_unique_reporters} unique users, {negative_user_pct:.1f}%)")

        # 7. Generate human-readable insights (no emojis)
        insights = []
        
        for topic, count in sorted(negative_reports.items(), key=lambda x: x[1], reverse=True):
            topic_info = NEGATIVE_TOPICS[topic]

            # Determine severity based on unique user count
            if count >= 10:
                prefix = "Many users have reported"
                suffix = "This issue seems widespread."
                severity = "widespread"
            elif count >= 5:
                prefix = "Several users mentioned"
                suffix = "It could be a recurring issue."
                severity = "common"
            elif count >= 2:
                prefix = "A few users noticed"
                suffix = "May not affect everyone."
                severity = "some"
            else:  # count == 1
                prefix = "One user reported"
                suffix = "This may be an isolated incident."
                severity = "few"

            insights.append({
                "text": f"{prefix} {topic_info['desc']}. {suffix}",
                "severity": severity,
                "category": topic
            })

        # 8. Summary message (no emojis)
        if warning_level == "high":
            summary = f"HIGH RISK: {total_negative_users} of {total_unique_reporters} unique users reported major issues."
        elif warning_level == "medium":
            summary = f"Moderate Risk: Some users reported recurring issues. Review before consumption."
        elif warning_level == "low":
            summary = f"Low Risk: A few isolated reports exist, but overall community sentiment is neutral."
        elif positive_count > 0:
            if positive_count >= total_unique_reporters * 0.7:
                summary = f"Mostly Positive: Majority of users reported a good experience."
            else:
                summary = f"Mixed Reviews: Some users were satisfied, others had minor complaints."
        else:
            summary = f"No significant feedback trends detected."

        # 9. Final structured result
        result = {
            "total_reports": total_reports_count,
            "total_unique_reporters": total_unique_reporters,
            "negative_reports": total_negative_users,
            "positive_reports": positive_count,
            "insights": insights,
            "product_name": product_name,
            "warning_level": warning_level,
            "summary_message": summary
        }

        print(f"DEBUG: Final result for {product_name}: {result}")
        return result

    except Exception as e:
        import traceback
        print(f"ERROR in get_community_warnings: {e}")
        traceback.print_exc()
        return {
            "total_reports": 0,
            "insights": [],
            "product_name": product_name,
            "warning_level": "none",
            "positive_reports": 0,
            "summary_message": "Error retrieving community warnings."
        }
# ==================== ROUTES ====================

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.errorhandler(429)
def ratelimit_handler(e):
    flash("Too many login attempts. Try again later.", "error")
    return redirect(url_for('landing'))

@limiter.limit("3 per minute", methods=["POST"])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return redirect(url_for('landing'))

    username = request.form['username']
    password = request.form['password']

    user = credentials_storage.get_user(username)

    if not user:
        flash("Account does not exist. Please sign up first.", "error")
        return redirect(url_for('landing'))

    hashed = user.get('password')

    if check_password_hash(hashed, password):
        session['username'] = username
        flash("Logged in successfully!", "success")
        return redirect(url_for('profile'))

    flash("Incorrect password.", "error")
    return redirect(url_for('landing'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    
    if credentials_storage.user_exists(username):
        flash("Username already exists.", "error")
        return redirect(url_for('landing'))

    if add_user(username, password):
        session['username'] = username
        flash("Account created successfully!", "success")
        return redirect(url_for('profile_form'))
    else:
        flash("Failed to create account. Please try again.", "error")
        return redirect(url_for('landing'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    username = session['username']
    scan_stats = scan_storage.get_user_scan_stats(username)
    
    return render_template('dashboard.html', 
                           username=username, 
                           scan_stats=scan_stats)

@app.route('/scan-label', methods=['GET', 'POST'])
def scan_label():
    if request.method == 'POST':
        if 'label_image' in request.files and request.files['label_image'].filename != '':
            file = request.files['label_image']
            scan_type = "label"
        elif 'barcode_image' in request.files and request.files['barcode_image'].filename != '':
            file = request.files['barcode_image']
            scan_type = "barcode"
        else:
            return render_template('scan_label.html', error="No file selected")

        if not allowed_file(file.filename):
            return render_template('scan_label.html', error="Only .jpg, .jpeg, .png files allowed")

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename).replace("\\", "/")

        try:
            image_stream = BytesIO(file.read())
            img = Image.open(image_stream)

            if img.format not in ("PNG", "JPEG"):
                return render_template('scan_label.html', error=f"Unsupported format: {img.format}. Please upload JPG or PNG.")

            img.verify()
            image_stream.seek(0)
            img = Image.open(image_stream).convert("RGB")
            img.save(filepath)

            if scan_type == "label":
                cleaned_text, structured_nutrients = process_label_image(filepath)

                session['scan_type'] = "label"
                session['scan_raw_text'] = cleaned_text
                session['scan_structured_nutrients'] = structured_nutrients
                session['scan_image_filename'] = unique_filename

                return redirect(url_for('scan_result'))

            elif scan_type == "barcode":
                processed_data = process_nutrition_label(filepath)

                session['scan_type'] = "barcode"
                session['barcode_processed'] = True
                session['scan_data'] = processed_data
                session['scan_image_filename'] = unique_filename

                return redirect(url_for('index'))

        except Exception as e:
            print(f"❌ Error while handling image: {e}")
            return render_template('scan_label.html', error=f"Could not process image: {e}")

    return render_template('scan_label.html')

@app.route('/scan-result')
def scan_result():
    scan_type = session.get('scan_type')
    
    if scan_type != "label":
        flash("No label scan data found.", "error")
        return redirect(url_for('scan_label'))
    
    raw_text = session.get('scan_raw_text')
    structured_nutrients = session.get('scan_structured_nutrients')
    username = session.get('username')

    if not raw_text or not structured_nutrients:
        return redirect(url_for('scan_label'))

    if not username:
        flash("Please log in to save scan results.", "error")
        return redirect(url_for('landing'))

    user_profile = load_user_health_profile(username)
    
    for nutrient in structured_nutrients:
        try:
            value_str = str(nutrient['value'])
            value = float(re.sub(r'[^\d.]', '', value_str))
        except (ValueError, TypeError):
            value = 0.0
            print(f"⚠️ Could not parse value for {nutrient.get('nutrient', 'Unknown')}: {nutrient.get('value', 'None')}")

        nutrient_name = nutrient.get('nutrient', '').strip()
        
        status_info = evaluate_nutrient_status_enhanced(nutrient_name, value)
        
        nutrient['value'] = value
        nutrient['status'] = status_info['status']
        nutrient['message'] = status_info['message']
        
        if user_profile:
            ai_analysis = get_personalized_nutrient_analysis(
                nutrient_name,
                value,
                nutrient.get('unit', 'g'),
                status_info['status'],
                user_profile
            )
            nutrient['ai_analysis'] = ai_analysis
        else:
            nutrient['ai_analysis'] = "Complete your profile to get personalized insights."
        
        print(f"🔍 Nutrient: {nutrient_name}, Value: {value}, Status: {status_info['status']}")

    try:
        image_filename = session.get('scan_image_filename')
        
        scan_id = scan_storage.save_scan_data(
            username=username,
            raw_text=raw_text,
            cleaned_text=raw_text,
            structured_nutrients=structured_nutrients,
            image_filename=image_filename,
            product_name="Scanned from Label (OCR)",
            product_image_url=None
        )
        
        if scan_id:
            session['last_scan_id'] = scan_id
            flash("Scan results saved successfully!", "success")
        else:
            flash("Could not save scan results to database.", "warning")
            
    except Exception as e:
        print(f"❌ Error saving scan to MongoDB: {e}")
        flash("Error saving scan results.", "warning")

    return render_template('scan_result.html', raw_text=raw_text, structured_nutrients=structured_nutrients)

@app.route('/index')
def index():
    barcode_processed = session.get('barcode_processed', False)
    scan_data = session.get('scan_data', {})
    username = session.get('username')
    
    # Initialize community_warnings
    community_warnings = {"total_reports": 0, "top_reports": [], "product_name": None}
    
    if barcode_processed and scan_data.get('structured_nutrients') and username:
        user_profile = load_user_health_profile(username)
        
        # Get community warnings for the product
        product_name = scan_data.get('product_name')
        if product_name:
            community_warnings = get_community_warnings(product_name)
            print(f"🔍 Community warnings for {product_name}: {community_warnings}")
        
        if user_profile:
            for nutrient in scan_data['structured_nutrients']:
                if 'ai_analysis' not in nutrient:
                    ai_analysis = get_personalized_nutrient_analysis(
                        nutrient.get('nutrient', ''),
                        nutrient.get('value', 0),
                        nutrient.get('unit', 'g'),
                        nutrient.get('status', 'Unknown'),
                        user_profile
                    )
                    nutrient['ai_analysis'] = ai_analysis

    return render_template(
        'index.html',
        barcode_processed=barcode_processed,
        scan_data=scan_data,
        community_warnings=community_warnings  # ← ADD THIS LINE
    )
@app.route('/add_to_diet', methods=['POST'])
def add_to_diet():
    username = session.get('username')
    scan_data = session.get('scan_data', {})
    image_filename = session.get('scan_image_filename')
    
    if not username:
        return jsonify({
            "status": "error",
            "message": "Please log in to save items to your diet."
        })
    
    product_name = scan_data.get('product_name')
    product_image_url = scan_data.get('product_image_url')
    structured_nutrients = scan_data.get('structured_nutrients', [])
    
    try:
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
            session.pop('scan_data', None)
            session.pop('barcode_processed', None)
            return jsonify({
                "status": "success",
                "message": "Item added to your diet successfully!"
            })
        else:
            return jsonify({
                "status": "warning",
                "message": "Could not add item to your diet."
            })
            
    except Exception as e:
        print(f"❌ Error saving barcode scan to MongoDB: {e}")
        return jsonify({
            "status": "error",
            "message": "Error saving item to your diet."
        })

@app.route('/my-scans')
def my_scans():
    if 'username' not in session:
        flash("Please log in to view your scans.", "error")
        return redirect(url_for('landing'))
    
    username = session['username']
    scans = scan_storage.get_user_scans(username)
    stats = scan_storage.get_user_scan_stats(username)
    
    return render_template('my_scans.html', scans=scans, stats=stats)

@app.route('/view-scan/<scan_id>')
def view_scan(scan_id):
    if 'username' not in session:
        flash("Please log in to view scans.", "error")
        return redirect(url_for('landing'))
    
    scan = scan_storage.get_scan_by_id(scan_id)
    
    if not scan or scan['username'] != session['username']:
        flash("Scan not found or access denied.", "error")
        return redirect(url_for('my_scans'))
    
    return render_template('view_scan.html', scan=scan)

@app.route('/delete-scan/<scan_id>', methods=['POST'])
def delete_scan(scan_id):
    if 'username' not in session:
        flash("Please log in.", "error")
        return redirect(url_for('landing'))
    
    username = session['username']
    
    if scan_storage.delete_scan(scan_id, username):
        flash("Scan deleted successfully.", "success")
    else:
        flash("Failed to delete scan.", "error")
    
    return redirect(url_for('my_scans'))

@app.route('/admin/cleanup-scans')
def cleanup_scans():
    deleted_count = scan_storage.cleanup_expired_scans()
    return jsonify({"message": f"Cleaned up {deleted_count} expired scans"})

@app.route('/debug/mongodb-collections')
def mongodb_collections():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    try:
        db = credentials_storage.mongo_config.db
        
        users_count = credentials_storage.collection.count_documents({}) if credentials_storage.collection else 0
        profiles_count = profile_storage.collection.count_documents({}) if profile_storage.collection else 0
        scans_count = scan_storage.collection.count_documents({}) if scan_storage.collection else 0
        
        html = f"""
        <h2>MongoDB Collections Status</h2>
        <div style="font-family: monospace; background: #f5f5f5; padding: 20px; border-radius: 5px;">
            <h3>Database: {db.name}</h3>
            <p><strong>User Credentials:</strong> {users_count} users</p>
            <p><strong>User Profiles:</strong> {profiles_count} profiles</p>
            <p><strong>Scan Data:</strong> {scans_count} scans</p>
            <hr>
            <p><strong>Collections in DB:</strong></p>
            <ul>
                {''.join([f'<li>{col}</li>' for col in db.list_collection_names()])}
            </ul>
        </div>
        <p><a href="{url_for('dashboard')}">← Back to Dashboard</a></p>
        """
        return html
        
    except Exception as e:
        return f"<h2>MongoDB Error</h2><p>Error: {e}</p>"

@app.route('/debug/my-profile-info')
def my_profile_info():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    username = session['username']
    profile = load_user_health_profile(username)
    
    if not profile:
        return "<h2>No Profile Found</h2><p>Please create your profile first.</p>"
    
    html = f"""
    <h2>Your Profile Information</h2>
    <div style="font-family: monospace; background: #f5f5f5; padding: 20px; border-radius: 5px;">
        <p><strong>Username:</strong> {profile.get('username', 'N/A')}</p>
        <p><strong>Full Name:</strong> {profile.get('full_name', 'N/A')}</p>
        <p><strong>Age:</strong> {profile.get('age', 'N/A')}</p>
        <p><strong>Gender:</strong> {profile.get('gender', 'N/A')}</p>
        <p><strong>Height:</strong> {profile.get('height_cm', 'N/A')} cm</p>
        <p><strong>Weight:</strong> {profile.get('weight_kg', 'N/A')} kg</p>
        <p><strong>Activity Level:</strong> {profile.get('activity_level', 'N/A')}</p>
        <p><strong>Medical Conditions:</strong> {profile.get('medical_conditions', 'None')}</p>
        <p><strong>Allergies:</strong> {profile.get('allergies', 'None')}</p>
        <p><strong>Created:</strong> {profile.get('created_at', 'N/A')}</p>
        <p><strong>Updated:</strong> {profile.get('updated_at', 'N/A')}</p>
    </div>
    <p><a href="{url_for('profile')}">← Back to Profile</a></p>
    """
    return html

@app.route('/profile-form', methods=['GET', 'POST'])
def profile_form():
    if 'username' not in session:
        return redirect(url_for('landing'))

    if request.method == 'POST':
        profile_data = {
            'username': session['username'],
            'full_name': request.form.get('full_name', ''),
            'height_cm': request.form.get('height_cm', ''),
            'weight_kg': request.form.get('weight_kg', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'activity_level': request.form.get('activity_level', ''),
            'medical_conditions': request.form.get('medical_conditions', ''),
            'allergies': request.form.get('allergies', ''),
        }
        
        if save_user_profile(profile_data):
            flash("Profile saved successfully!", "success")
            return redirect(url_for('profile'))
        else:
            flash("Failed to save profile. Please try again.", "error")

    return render_template('profile_form.html')

@app.route('/get-ai-analysis', methods=['POST'])
def get_ai_analysis():
    if 'username' not in session:
        return jsonify({"error": "Please log in to get analysis"}), 401
    
    try:
        analysis_html = get_comprehensive_ai_analysis()
        return jsonify({"success": True, "analysis": analysis_html})
    except Exception as e:
        print(f"❌ Error generating AI analysis: {e}")
        return jsonify({"error": "Failed to generate analysis"}), 500
@app.route('/food-tracker', methods=['GET', 'POST'])
def food_tracker():
    if 'username' not in session:
        return redirect(url_for('landing'))

    if 'food_items' not in session:
        session['food_items'] = []

    if request.method == 'POST':
        if 'food_image' in request.files:
            # Handle scan upload
            image = request.files['food_image']
            if image.filename != '':
                nutrients = process_label_image(image)  # Your OCR function
                if nutrients:
                    session['food_items'].append({
                        'item': 'Scanned Item',
                        'quantity': 1,
                        'protein': nutrients.get('protein', 0),
                        'carbs': nutrients.get('carbohydrates', 0),
                        'fats': nutrients.get('fats', 0)
                    })
                    session.modified = True
                    flash("Scanned food item added.")
        else:
            # Handle manual form entry
            item = request.form.get('item')
            quantity = float(request.form.get('quantity', 0))
            protein = float(request.form.get('protein', 0))
            carbs = float(request.form.get('carbs', 0))
            fats = float(request.form.get('fats', 0))

            # Check if item already exists, update if so
            updated = False
            for food in session['food_items']:
                if food['item'].lower() == item.lower():
                    food['quantity'] += quantity
                    food['protein'] += protein
                    food['carbs'] += carbs
                    food['fats'] += fats
                    updated = True
                    break
            if not updated:
                session['food_items'].append({
                    'item': item,
                    'quantity': quantity,
                    'protein': protein,
                    'carbs': carbs,
                    'fats': fats
                })
            session.modified = True
            flash("Food item added/updated.")

    # Calculate totals
    total = {
        'protein': sum(item['protein'] for item in session['food_items']),
        'carbs': sum(item['carbs'] for item in session['food_items']),
        'fats': sum(item['fats'] for item in session['food_items']),
    }

    return render_template('food_tracker.html', food_items=session['food_items'], total=total)

@app.route('/get-health-score-data/<period>')
def get_health_score_data(period='weekly'):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = session['username']
    profile = load_user_health_profile(username)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
        
    if not all(k in profile and profile[k] for k in ['weight_kg', 'activity_level']):
        return jsonify({"error": "Profile incomplete"}), 400

    valid_periods = ['daily', 'weekly', 'monthly']
    if period not in valid_periods:
        return jsonify({"error": "Invalid period"}), 400

    recommendations = calculate_daily_needs(profile['weight_kg'], profile['activity_level'])
    data = get_historical_health_scores(username, period, recommendations)
    
    goal_score = 85 
    
    return jsonify({
        "labels": data["labels"],
        "actual_scores": data["scores"],
        "goal_scores": [goal_score] * len(data["labels"])
    })

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    username = session['username']
    profile_data = load_user_health_profile(username)
    if not profile_data:
        flash("Your Profile Doesn't exist, Create a profile", "warning")
        return redirect(url_for('profile_form'))

    try:
        recommendation = calculate_daily_needs(profile_data['weight_kg'], profile_data['activity_level'])
    except (ValueError, KeyError):
        flash("Your profile is incomplete. Please update your weight and activity level.", "warning")
        return redirect(url_for('edit_profile'))

    today_scans_data = get_today_scan_visualization()
    
    today_totals = {"protein": 0, "carbs": 0, "fats": 0}
    for scan in today_scans_data:
        for nutrient_name, nutrient_data in scan["nutrients"].items():
            if nutrient_name in ["protein"]:
                today_totals["protein"] += nutrient_data["value"]
            elif nutrient_name in ["carbohydrates", "carbs"]:
                today_totals["carbs"] += nutrient_data["value"]
            elif nutrient_name in ["fat", "fats"]:
                today_totals["fats"] += nutrient_data["value"]

    percent = lambda val, ref: round((val / ref) * 100) if ref else 0
    percentages = {
        'protein': percent(today_totals['protein'], recommendation['protein_g']),
        'carbs': percent(today_totals['carbs'], recommendation['carbs_g']),
        'fats': percent(today_totals['fats'], recommendation['fats_g'])
    }

    scan_stats = scan_storage.get_user_scan_stats(username)

    return render_template(
        'profile.html', 
        profile=profile_data, 
        intake={
            'protein_g': round(today_totals['protein'], 1),
            'carbs_g': round(today_totals['carbs'], 1),
            'fats_g': round(today_totals['fats'], 1)
        }, 
        recommendation=recommendation, 
        percentages=percentages,
        today_scans=today_scans_data,
        scan_stats=scan_stats
    )

@app.route('/get-scan-count-data/<period>')
def get_scan_count_data(period='weekly'):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = session['username']
    
    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    valid_periods = ['daily', 'weekly', 'monthly']
    if period not in valid_periods:
        return jsonify({"error": "Invalid period"}), 400

    end_date = datetime.utcnow()
    
    if period == 'daily':
        start_date = end_date - timedelta(days=30)
        date_format = "%Y-%m-%d"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
    elif period == 'weekly':
        start_date = end_date - timedelta(weeks=12)
        date_format = "%Y-W%U"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
    else:
        start_date = end_date - timedelta(days=365)
        date_format = "%Y-%m"
        group_id = {"$dateToString": {"format": date_format, "date": "$scan_date"}}

    try:
        pipeline = [
            {
                "$match": {
                    "username": username,
                    "scan_date": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": group_id,
                    "scan_count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]

        results = list(scan_storage.collection.aggregate(pipeline))
        
        labels = [result['_id'] for result in results]
        scan_counts = [result['scan_count'] for result in results]

        formatted_labels = []
        for label in labels:
            if period == 'weekly':
                try:
                    year, week = label.split('-W')
                    formatted_labels.append(f"Week {week}, {year}")
                except:
                    formatted_labels.append(label)
            elif period == 'monthly':
                try:
                    year, month = label.split('-')
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    formatted_labels.append(f"{month_names[int(month)-1]} {year}")
                except:
                    formatted_labels.append(label)
            else:
                formatted_labels.append(label)

        return jsonify({
            "labels": formatted_labels,
            "scan_counts": scan_counts,
            "total_scans": sum(scan_counts)
        })

    except Exception as e:
        print(f"❌ Error fetching scan count data: {e}")
        return jsonify({"error": "Failed to fetch scan data"}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('landing'))

@app.route('/download-report')
def download_report():
    return "Download functionality not implemented yet."

@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    username = session['username']
    
    if request.method == 'POST':
        profile_data = {
            'username': username,
            'full_name': request.form.get('full_name', ''),
            'height_cm': request.form.get('height_cm', ''),
            'weight_kg': request.form.get('weight_kg', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'activity_level': request.form.get('activity_level', ''),
            'medical_conditions': request.form.get('medical_conditions', ''),
            'allergies': request.form.get('allergies', ''),
        }
        
        if save_user_profile(profile_data):
            flash("Profile updated successfully!", "success")
            return redirect(url_for('profile'))
        else:
            flash("Failed to update profile. Please try again.", "error")

    profile = load_user_health_profile(username)
    if not profile:
        flash("No profile found. Please create one.", "warning")
        return redirect(url_for('profile_form'))

    return render_template('edit_profile.html', profile=profile)

@app.route('/delete-account', methods=['POST'])
def delete_account():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session['username']
    
    try:
        credentials_storage.collection.delete_one({"username": username})
        profile_storage.delete_profile(username)
        scan_storage.collection.delete_many({"username": username})
        session.clear()
        
        return jsonify({
            "success": True,
            "message": "Account deleted successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
# Add these routes to your app.py file
# Add these routes to your app.py file

@app.route('/view-all-scans')
def view_all_scans():
    """Display the view scans page"""
    if 'username' not in session:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            # If it's an AJAX request, return JSON instead of redirect
            return jsonify({"error": "Unauthorized"}), 401
        flash("Please log in to view your scans.", "error")
        return redirect(url_for('landing'))
    
    return render_template('view_all_scans.html')


@app.route('/api/get-all-scans')
def api_get_all_scans():
    """API endpoint to fetch all scans for the logged-in user"""
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session['username']

    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    try:
        # Fetch all scans for the user
        scans = list(scan_storage.collection.find(
            {"username": username}
        ).sort("scan_date", -1))

        # Convert ObjectId and datetime
        for scan in scans:
            scan['_id'] = str(scan['_id'])
            if isinstance(scan.get('scan_date'), datetime):
                scan['scan_date'] = scan['scan_date'].isoformat()

        # Calculate statistics
        total_scans = len(scans)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_scans = len([
            s for s in scans
            if 'scan_date' in s and datetime.fromisoformat(s['scan_date']) >= week_ago
        ])

        # Count high risk items
        high_risk_count = 0
        for scan in scans:
            summary = scan.get('nutrition_analysis', {}).get('summary', {})
            high_risk_count += summary.get('high_risk_count', 0)

        stats = {
            "total_scans": total_scans,
            "recent_scans": recent_scans,
            "high_risk_count": high_risk_count
        }

        return jsonify({
            "success": True,
            "scans": scans,
            "stats": stats
        })

    except Exception as e:
        print(f"❌ Error fetching scans: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": "Failed to fetch scans"}), 500


@app.route('/api/delete-scan/<scan_id>', methods=['DELETE'])
def api_delete_scan(scan_id):
    """API endpoint to delete a specific scan"""
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = session['username']

    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    try:
        # Validate ObjectId
        try:
            obj_id = ObjectId(scan_id)
        except (InvalidId, TypeError):
            print(f"❌ Invalid scan ID format: {scan_id}")
            return jsonify({
                "success": False,
                "error": "Invalid scan ID format"
            }), 400

        # Perform deletion
        result = scan_storage.collection.delete_one({
            "_id": obj_id,
            "username": username
        })

        if result.deleted_count > 0:
            print(f"✅ Scan deleted: {scan_id} by {username}")
            return jsonify({
                "success": True,
                "message": "Scan deleted successfully"
            })
        else:
            print(f"⚠️ Unauthorized or missing scan: {scan_id}")
            return jsonify({
                "success": False,
                "error": "Scan not found or you don't have permission to delete it"
            }), 404

    except Exception as e:
        print(f"❌ Error deleting scan: {e}")
        import traceback; traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500
# Place this near your other API routes like api_delete_scan

@app.route('/api/submit-report/<scan_id>', methods=['POST'])
def api_submit_report(scan_id):
    """API endpoint to submit an issue report for a scan"""
    if 'username' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    username = session['username']

    if scan_storage.collection is None:
        return jsonify({"success": False, "error": "Database not available"}), 503

    try:
        # Validate ObjectId
        try:
            obj_id = ObjectId(scan_id)
        except (InvalidId, TypeError):
            print(f"❌ Invalid scan ID format for report: {scan_id}")
            return jsonify({"success": False, "error": "Invalid scan ID format"}), 400

        # Get report data from JSON body
        report_data = request.get_json()
        if not report_data or 'description' not in report_data or not report_data['description']:
            return jsonify({"success": False, "error": "Report description is required"}), 400

        description = report_data['description'].strip()
        severity = report_data.get('severity') # Optional
        contact_consent = report_data.get('contact_consent', False) # Optional, defaults to False

        # Prepare the update document
        report_payload = {
            "reported_at": datetime.utcnow(),
            "description": description,
            "severity": severity,
            "contact_consent": contact_consent,
            "status": "submitted" # You can add status tracking later
        }

        # Update the specific scan document
        # We use $set to add or update the 'user_feedback.issue_report' field
        result = scan_storage.collection.update_one(
            {"_id": obj_id, "username": username}, # Ensure user owns the scan
            {"$set": {"user_feedback.issue_report": report_payload}} 
            # Note: This overwrites previous reports in this structure.
            # To store multiple reports, use '$push' with an array field instead:
            # {"$push": {"user_feedback.issue_reports": report_payload}}
        )

        if result.matched_count > 0:
            if result.modified_count > 0:
                 print(f"✅ Issue report added/updated for scan: {scan_id} by {username}")
                 return jsonify({"success": True, "message": "Report submitted successfully"})
            else:
                 # Matched but nothing changed (e.g., submitted identical report twice)
                 print(f"ℹ️ Issue report submitted but no changes detected for scan: {scan_id}")
                 return jsonify({"success": True, "message": "Report received (no changes detected)"})
        else:
            print(f"⚠️ Report submission failed: Scan not found or unauthorized for ID {scan_id}")
            return jsonify({"success": False, "error": "Scan not found or access denied"}), 404

    except Exception as e:
        print(f"❌ Error submitting report for scan {scan_id}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

