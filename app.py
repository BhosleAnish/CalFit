from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import csv, os, re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ocr_utils import process_label_image,extract_nutrients
from PIL import Image
import pillow_avif  # enables AVIF support in Pillow
from io import BytesIO
import sqlite3
import pymongo
from bson import ObjectId
from process_label import process_nutrition_label
import google.generativeai as genai
from datetime import datetime, timedelta
import openai
import json
load_dotenv()

# --- CONFIGURE THE AI MODEL ---
# This loads the key from your .env file
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Instantiate the client
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

csrf = CSRFProtect(app)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

# File paths
USER_CSV = 'users.csv'
PROFILE_CSV = 'user_profiles.csv'
SCAN_CSV = 'user_scans.csv'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize CSVs if not exist
for file, headers in {
    USER_CSV: ['username', 'password'],
    PROFILE_CSV: ['username', 'full_name', 'age', 'gender', 'height_cm', 'weight_kg', 'activity_level', 'medical_conditions', 'allergies'],
    SCAN_CSV: ['username', 'item', 'quantity', 'protein', 'carbs', 'fats', 'date']
}.items():
    if not os.path.exists(file):
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

# MongoDB Configuration
class MongoConfig:
    def __init__(self):
        # MongoDB connection string - replace with your actual connection details
        self.MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.DATABASE_NAME = os.getenv('MONGO_DB_NAME', 'nutrition_app')
        self.COLLECTION_NAME = 'user_scans'
        
        try:
            # Connect to MongoDB Atlas with additional options
            self.client = pymongo.MongoClient(
                self.MONGO_URI,
                retryWrites=True,
                w='majority',
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            
            self.db = self.client[self.DATABASE_NAME]
            self.collection = self.db[self.COLLECTION_NAME]
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ MongoDB connection successful")
            print(f"✅ Connected to database: {self.DATABASE_NAME}")
            
            # Create indexes for better performance and automatic cleanup
            self.setup_indexes()
            
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"❌ MongoDB connection timeout: {e}")
            self.client = None
            self.db = None
            self.collection = None
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def setup_indexes(self):
        """Create indexes for better performance and automatic cleanup"""
        try:
            # Index for faster user queries
            self.collection.create_index("username")
            
            # Index for date-based queries
            self.collection.create_index("scan_date")
            
            # Compound index for user + date queries
            self.collection.create_index([("username", 1), ("scan_date", -1)])
            
            # TTL (Time To Live) index for automatic cleanup after 30 days
            self.collection.create_index("expires_at", expireAfterSeconds=0)
            
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"⚠️ Could not create indexes: {e}")

# MongoDB utility functions
class ScanStorage:
    def __init__(self):
        self.mongo_config = MongoConfig()
        self.collection = self.mongo_config.collection
    
    def save_scan_data(self, username, raw_text, cleaned_text, structured_nutrients, image_filename=None, product_name=None, product_image_url=None):
        """Save scan data to MongoDB"""
        if self.collection is None:
            print("❌ MongoDB not available, cannot save scan data")
            return None
        
        try:
            # Create nutrition summary
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
        """Create a summary of nutrition data for easier viewing in Atlas"""
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
        
        # Group by categories
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
                sort=[("scan_date", -1)]  # Most recent first
            ).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
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
        """Delete a specific scan (with username verification for security)"""
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
            # Total scans
            total_scans = self.collection.count_documents({"username": username})
            
            # Scans in last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_scans = self.collection.count_documents({
                "username": username,
                "scan_date": {"$gte": week_ago}
            })
            
            # Most recent scan date
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
        """Remove expired scans (older than 30 days)"""
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

# Initialize scan storage
scan_storage = ScanStorage()

# Utility functions
def load_users():
    users = {}
    with open(USER_CSV, 'r') as f:
        for row in csv.DictReader(f):
            users[row['username']] = row['password']
    return users

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def add_user(username, password):
    hashed_password = generate_password_hash(password)
    with open(USER_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([username, hashed_password])

def save_user_profile(data):
    fieldnames = ['username', 'full_name', 'height_cm', 'weight_kg', 'age', 'gender', 'activity_level', 'medical_conditions', 'allergies']
    profiles = []
    with open(PROFILE_CSV, 'r') as f:
        profiles = [row for row in csv.DictReader(f) if row['username'] != data['username']]
    profiles.append(data)
    with open(PROFILE_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profiles)

def load_user_health_profile(username):
    with open(PROFILE_CSV, 'r') as f:
        for row in csv.DictReader(f):
            if row['username'] == username:
                return row
    return None

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

def load_user_scans(username):
    if not os.path.exists(SCAN_CSV):
        return []
    with open(SCAN_CSV, 'r') as f:
        return [row for row in csv.DictReader(f) if row['username'] == username]

def get_weekly_summary(scans):
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    weekly = [s for s in scans if datetime.strptime(s['date'], '%Y-%m-%d') >= week_ago]
    total = lambda key: sum(float(s.get(key, 0)) for s in weekly)
    count = len(weekly)
    return {
        'avg_protein': round(total('protein') / count, 1) if count else 0,
        'avg_carbs': round(total('carbs') / count, 1) if count else 0,
        'avg_fats': round(total('fats') / count, 1) if count else 0
    }

def evaluate_nutrient_status(nutrient_name, value):
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
        FROM ingredient_thresholds 
        WHERE LOWER(name) = LOWER(?)
    """, (nutrient_name,))
    
    result = cursor.fetchone()
    conn.close()

    if not result:
        return {"status": "Unknown", "message": "No data available."}

    max_threshold, min_threshold, high_msg, low_msg = result

    if max_threshold is not None and value > max_threshold:
        return {"status": "High", "message": high_msg}
    elif min_threshold is not None and value < min_threshold:
        return {"status": "Low", "message": low_msg}
    else:
        return {"status": "Normal", "message": "This level is within the healthy range."}

# Enhanced evaluate_nutrient_status function
def evaluate_nutrient_status_enhanced(nutrient_name, value):
    """Enhanced function to evaluate nutrient status with better name matching"""
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()

    # First, try exact match (case insensitive)
    cursor.execute("""
        SELECT max_threshold, min_threshold, high_risk_message, low_risk_message 
        FROM ingredient_thresholds 
        WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
    """, (nutrient_name,))
    
    result = cursor.fetchone()
    
    # If no exact match, try partial matching for common variations
    if not result:
        # Common nutrient name mappings
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
        
        # Try to find a match using mappings
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
    
    # If still no match, try LIKE search
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

    # Evaluate status based on thresholds
    if max_threshold is not None and value > max_threshold:
        return {"status": "High", "message": high_msg}
    elif min_threshold is not None and value < min_threshold:
        return {"status": "Low", "message": low_msg}
    else:
        return {"status": "Normal", "message": "This level is within the healthy range."}


def get_user_habit_summary(username, days=30):
    """
    Aggregates a user's scan history from MongoDB to create a 30-day habit summary.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    pipeline = [
        {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
        {"$unwind": "$nutrition_analysis.structured_nutrients"},
        {
            "$group": {
                "_id": "$nutrition_analysis.structured_nutrients.nutrient",
                "total_value": {"$sum": "$nutrition_analysis.structured_nutrients.value"},
                "count": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$nutrition_analysis.structured_nutrients.status", "High"]},
                            1, 0
                        ]
                    }
                },
                "avg_value": {"$avg": "$nutrition_analysis.structured_nutrients.value"},
                "unit": {"$first": "$nutrition_analysis.structured_nutrients.unit"},
                "category": {"$first": "$nutrition_analysis.structured_nutrients.category"}
            }
        },
        {"$sort": {"high_risk_count": -1, "total_value": -1}},
        {"$limit": 10}
    ]
    
    try:
        summary = list(scan_storage.collection.aggregate(pipeline))
        total_scans = scan_storage.collection.count_documents(
            {"username": username, "scan_date": {"$gte": start_date}}
        )
        
        # Reformat results
        for item in summary:
            item['nutrient'] = item.pop('_id')
            item['avg_value_per_scan'] = round(item['avg_value'], 2)
            item['total_value'] = round(item['total_value'], 2)

        return {"total_scans": total_scans, "nutrient_summary": summary}
    except Exception as e:
        print(f"❌ Error during habit aggregation: {e}")
        return None
def get_today_scan_visualization():
    """
    Get today's scans for visualization on profile page
    """
    username = session.get("username")
    if not username or scan_storage.collection is None:  # Fixed: use 'is None' instead of 'not'
        return []

    try:
        # Get today's date range
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        # Fetch today's scans
        today_scans = list(scan_storage.collection.find({
            "username": username,
            "scan_date": {
                "$gte": today_start,
                "$lt": today_end
            }
        }).sort("scan_date", -1))

        # Process scans for visualization
        visualization_data = []
        for scan in today_scans:
            product_name = scan.get("product_info", {}).get("name", "Unknown Product")
            nutrients = scan.get("nutrition_analysis", {}).get("structured_nutrients", [])
            
            # Extract key nutrients for visualization
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
    """
    Generates a personalized, doctor-style analysis of eating habits 
    using all scans saved by the logged-in user and their profile data.
    """
    username = session.get("username")
    if not username:
        return "<h2>Error</h2><p>No user logged in.</p>"

    if scan_storage.collection is None:  # Fixed: use 'is None' instead of 'not'
        return "<h2>AI Analysis Unavailable</h2><p>Database not connected.</p>"

    try:
        # Fetch all scans for this user
        user_scans = list(scan_storage.collection.find({"username": username}))
        
        # Get user profile
        user_profile = load_user_health_profile(username)
        
        if not user_profile:
            return "<h2>Profile Required</h2><p>Please complete your profile to get personalized analysis.</p>"
            
    except Exception as e:
        print(f"❌ Could not fetch data: {e}")
        return "<h2>Error</h2><p>Could not fetch user data.</p>"

    if not user_scans:
        return "<h2>No Data</h2><p>No scans found for your profile. Start scanning some food items!</p>"

    # Aggregate nutritional data
    total_nutrients = {}
    risk_counts = {"High": 0, "Low": 0, "Normal": 0, "Unknown": 0}
    product_names = []
    scan_count = len(user_scans)
    
    for scan in user_scans:
        # Collect product names
        product_name = scan.get("product_info", {}).get("name", "Unknown")
        if product_name != "Unknown":
            product_names.append(product_name)
        
        # Process nutrients
        structured_nutrients = scan.get("nutrition_analysis", {}).get("structured_nutrients", [])
        for nutrient in structured_nutrients:
            nutrient_name = nutrient.get("nutrient", "").lower()
            value = nutrient.get("value", 0)
            status = nutrient.get("status", "Unknown")
            
            # Aggregate nutrient values
            if nutrient_name:
                if nutrient_name not in total_nutrients:
                    total_nutrients[nutrient_name] = {"total": 0, "count": 0, "unit": nutrient.get("unit", "")}
                total_nutrients[nutrient_name]["total"] += value
                total_nutrients[nutrient_name]["count"] += 1
            
            # Count risk statuses
            if status in risk_counts:
                risk_counts[status] += 1

    # Calculate averages
    avg_nutrients = {}
    for nutrient, data in total_nutrients.items():
        if data["count"] > 0:
            avg_nutrients[nutrient] = {
                "average": round(data["total"] / data["count"], 2),
                "unit": data["unit"],
                "total": round(data["total"], 2)
            }

    # Prepare data for AI prompt
    analysis_data = {
        "user_profile": {
            "name": user_profile.get("full_name", "User"),
            "age": user_profile.get("age", "Not specified"),
            "gender": user_profile.get("gender", "Not specified"),
            "height_cm": user_profile.get("height_cm", "Not specified"),
            "weight_kg": user_profile.get("weight_kg", "Not specified"),
            "activity_level": user_profile.get("activity_level", "Not specified"),
            "medical_conditions": user_profile.get("medical_conditions", "None specified"),
            "allergies": user_profile.get("allergies", "None specified")
        },
        "scan_summary": {
            "total_scans": scan_count,
            "products_scanned": list(set(product_names)),
            "risk_distribution": risk_counts,
            "average_nutrients": avg_nutrients
        }
    }

    # Calculate daily nutritional needs based on profile
    if user_profile.get("weight_kg") and user_profile.get("activity_level"):
        try:
            daily_needs = calculate_daily_needs(user_profile["weight_kg"], user_profile["activity_level"])
            analysis_data["daily_recommendations"] = daily_needs
        except Exception as e:
            print(f"Warning: Could not calculate daily needs: {e}")

    data_json = json.dumps(analysis_data, indent=2)

    # Doctor-style AI prompt
    prompt = f"""
    You are a qualified nutritionist and dietitian providing a comprehensive health assessment. 
    Analyze the following complete dietary data for your patient:

    ```json
    {data_json}
    ```

    Provide a thorough, personalized analysis in HTML format with these sections:

    <h3>Patient Overview</h3>
    - Summarize key demographic and health information
    - Note any medical conditions or allergies that affect dietary recommendations

    <h3>Dietary Pattern Analysis</h3>
    - Analyze the types of products frequently consumed
    - Identify eating patterns and food choices
    - Comment on diet diversity and balance

    <h3>Nutritional Assessment</h3>
    - Evaluate key nutrients (protein, carbs, fats, sodium, sugar, fiber)
    - Compare intake patterns to recommended daily values for this individual
    - Highlight both deficiencies and excesses

    <h3>Health Risks Identified</h3>
    - List specific health risks based on the nutritional data
    - Connect risks to the patient's profile (age, activity level, medical conditions)
    - Prioritize risks by severity

    <h3>Positive Aspects</h3>
    - Acknowledge any healthy choices or balanced nutrients
    - Recognize good dietary habits observed

    <h3>Personalized Recommendations</h3>
    - Provide specific, actionable dietary changes
    - Suggest foods to increase or decrease
    - Consider the patient's lifestyle and preferences
    - Include portion size guidance where relevant

    <h3>Action Plan</h3>
    - Create a prioritized list of 3-5 key changes to implement
    - Suggest a timeline for implementing changes
    - Recommend follow-up monitoring

    Keep the tone professional but compassionate. Be direct about health risks while providing constructive guidance. Focus on sustainable, realistic changes.
    """

    try:
        if not client:
            return "<h2>AI Analysis Unavailable</h2><p>OpenAI client not configured.</p>"
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a qualified nutritionist and dietitian providing comprehensive health assessments based on dietary data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return "<h2>Analysis Error</h2><p>Could not generate AI analysis. Please try again later.</p>"


# ---------------- ROUTES ---------------- #

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
    users = load_users()

    # Check if username exists
    if username not in users:
        flash("Account does not exist. Please sign up first.", "error")
        return redirect(url_for('landing'))

    hashed = users.get(username)

    if check_password_hash(hashed, password):
        session['username'] = username
        flash("Logged in successfully!", "success")
        return redirect(url_for('profile'))

    flash("Incorrect password.", "error")
    return redirect(url_for('landing'))

@app.route('/signup', methods=['POST'])
def signup():
    username, password = request.form['username'], request.form['password']
    if username in load_users():
        flash("Username already exists.", "error")
        return redirect(url_for('landing'))

    add_user(username, password)
    session['username'] = username
    return redirect(url_for('profile_form'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    username = session['username']
    
    # Get scan statistics for dashboard
    scan_stats = scan_storage.get_user_scan_stats(username)
    
    return render_template('dashboard.html', 
                           username=username, 
                           scan_stats=scan_stats)

@app.route('/scan-label', methods=['GET', 'POST'])
def scan_label():
    if request.method == 'POST':
        # --- Case 1: Food Label OCR ---
        if 'label_image' in request.files and request.files['label_image'].filename != '':
            file = request.files['label_image']
            scan_type = "label"
        # --- Case 2: Barcode Image ---
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
            # Save the uploaded image
            image_stream = BytesIO(file.read())
            img = Image.open(image_stream)

            if img.format not in ("PNG", "JPEG"):
                return render_template('scan_label.html', error=f"Unsupported format: {img.format}. Please upload JPG or PNG.")

            img.verify()
            image_stream.seek(0)
            img = Image.open(image_stream).convert("RGB")
            img.save(filepath)

            # --- Decide which form was used and redirect accordingly ---
            if scan_type == "label":
                # OCR flow - redirect to scan_result
                cleaned_text, structured_nutrients = process_label_image(filepath)

                session['scan_type'] = "label"
                session['scan_raw_text'] = cleaned_text
                session['scan_structured_nutrients'] = structured_nutrients
                session['scan_image_filename'] = unique_filename

                return redirect(url_for('scan_result'))

            elif scan_type == "barcode":
                # Barcode flow - redirect to index
                processed_data = process_nutrition_label(filepath)

                session['scan_type'] = "barcode"
                session['barcode_processed'] = True
                session['scan_data'] = processed_data
                session['scan_image_filename'] = unique_filename

                return redirect(url_for('index'))  # Redirect to index.html instead

        except Exception as e:
            print(f"❌ Error while handling image: {e}")
            return render_template('scan_label.html', error=f"Could not process image: {e}")

    return render_template('scan_label.html')


@app.route('/scan-result')
def scan_result():
    # This route now only handles label scans
    scan_type = session.get('scan_type')
    
    # Ensure this is a label scan, not barcode
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

    # Evaluate each nutrient's status and risk message properly
    for nutrient in structured_nutrients:
        try:
            # Clean the value - remove any non-numeric characters except decimal points
            value_str = str(nutrient['value'])
            value = float(re.sub(r'[^\d.]', '', value_str))
        except (ValueError, TypeError):
            value = 0.0
            print(f"⚠️ Could not parse value for {nutrient.get('nutrient', 'Unknown')}: {nutrient.get('value', 'None')}")

        nutrient_name = nutrient.get('nutrient', '').strip()
        
        # Get status evaluation from database using enhanced function
        status_info = evaluate_nutrient_status_enhanced(nutrient_name, value)
        
        # Update nutrient info
        nutrient['value'] = value  # Clean numeric value
        nutrient['status'] = status_info['status']
        nutrient['message'] = status_info['message']
        
        # Debug output
        print(f"🔍 Nutrient: {nutrient_name}, Value: {value}, Status: {status_info['status']}")

    # Save scan data to MongoDB
    try:
        image_filename = session.get('scan_image_filename')
        
        scan_id = scan_storage.save_scan_data(
            username=username,
            raw_text=raw_text,
            cleaned_text=raw_text,  # Using raw_text as cleaned_text since cleaned_text isn't set
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

    return render_template(
        'index.html',
        barcode_processed=barcode_processed,
        scan_data=scan_data
    )

@app.route('/add_to_diet', methods=['POST'])
def add_to_diet():
    username = session.get('username')
    scan_data = session.get('scan_data', {})
    image_filename = session.get('scan_image_filename')

    if not username:
        flash("Please log in to save items to your diet.", "error")
        return redirect(url_for('index'))

    # Extract data from the standardized dictionary
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
            flash("Item added to your diet successfully!", "success")
        else:
            flash("Could not add item to your diet.", "warning")
    except Exception as e:
        print(f"❌ Error saving barcode scan to MongoDB: {e}")
        flash("Error saving item to your diet.", "error")

    # Clear session data after saving
    session.pop('scan_data', None)
    session.pop('barcode_processed', None)

    return redirect(url_for('index'))

@app.route('/my-scans')
def my_scans():
    """View all user's saved scans"""
    if 'username' not in session:
        flash("Please log in to view your scans.", "error")
        return redirect(url_for('landing'))
    
    username = session['username']
    scans = scan_storage.get_user_scans(username)
    stats = scan_storage.get_user_scan_stats(username)
    
    return render_template('my_scans.html', scans=scans, stats=stats)

@app.route('/view-scan/<scan_id>')
def view_scan(scan_id):
    """View a specific scan by ID"""
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
    """Delete a specific scan"""
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
    """Admin route to manually trigger scan cleanup"""
    deleted_count = scan_storage.cleanup_expired_scans()
    return jsonify({"message": f"Cleaned up {deleted_count} expired scans"})

@app.route('/debug/mongodb-info')
def mongodb_info():
    """Debug route to check MongoDB connection and stats"""
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    if scan_storage.collection is None:
        return "<h2>MongoDB Status: Not Connected</h2><p>Check your MONGO_URI in .env file</p>"
    
    try:
        # Get connection stats
        server_info = scan_storage.mongo_config.client.server_info()
        db_stats = scan_storage.mongo_config.db.command("dbstats")
        
        total_scans = scan_storage.collection.count_documents({})
        user_scans = scan_storage.collection.count_documents({"username": session['username']})
        
        html = f"""
        <h2>MongoDB Atlas Connection Info</h2>
        <div style="font-family: monospace; background: #f5f5f5; padding: 20px; border-radius: 5px;">
            <p><strong>Status:</strong> ✅ Connected</p>
            <p><strong>Database:</strong> {scan_storage.mongo_config.DATABASE_NAME}</p>
            <p><strong>Collection:</strong> {scan_storage.mongo_config.COLLECTION_NAME}</p>
            <p><strong>Server Version:</strong> {server_info.get('version', 'N/A')}</p>
            <p><strong>Total Scans:</strong> {total_scans}</p>
            <p><strong>Your Scans:</strong> {user_scans}</p>
            <p><strong>Database Size:</strong> {db_stats.get('dataSize', 0)} bytes</p>
        </div>
        <p><a href="{url_for('dashboard')}">← Back to Dashboard</a></p>
        """
        
        return html
        
    except Exception as e:
        return f"<h2>MongoDB Error</h2><p>Error: {e}</p>"

@app.route('/debug-thresholds')
def debug_thresholds():
    """Debug route to see what's in your thresholds database"""
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, max_threshold, min_threshold FROM ingredient_thresholds ORDER BY name")
    results = cursor.fetchall()
    conn.close()
    
    html = "<h2>Database Thresholds</h2><table border='1'><tr><th>Name</th><th>Max</th><th>Min</th></tr>"
    for row in results:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    html += "</table>"
    return html

@app.route('/profile-form', methods=['GET', 'POST'])
def profile_form():
    if 'username' not in session:
        return redirect(url_for('landing'))

    if request.method == 'POST':
        save_user_profile({
            'username': session['username'],
            'full_name': request.form['full_name'],
            'height_cm': request.form['height_cm'],
            'weight_kg': request.form['weight_kg'],
            'age': request.form['age'],
            'gender': request.form['gender'],
            'activity_level': request.form['activity_level'],
            'medical_conditions': request.form['medical_conditions'],
            'allergies': request.form['allergies'],
        })
        return redirect(url_for('profile'))

    return render_template('profile_form.html')

# Add new route for AI analysis
@app.route('/get-ai-analysis', methods=['POST'])
def get_ai_analysis():
    """Route to generate and return AI analysis"""
    if 'username' not in session:
        return jsonify({"error": "Please log in to get analysis"}), 401
    
    try:
        analysis_html = get_comprehensive_ai_analysis()
        return jsonify({"success": True, "analysis": analysis_html})
    except Exception as e:
        print(f"❌ Error generating AI analysis: {e}")
        return jsonify({"error": "Failed to generate analysis"}), 500


# Update the existing profile route
@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    username = session['username']
    profile = load_user_health_profile(username)
    if not profile:
        flash("Your Profile Doesn't exist, Create a profile")
        return redirect(url_for('profile_form'))

    # Calculate daily recommendations
    recommendation = calculate_daily_needs(profile['weight_kg'], profile['activity_level'])
    
    # Get today's scan data for visualization
    today_scans_data = get_today_scan_visualization()
    
    # Calculate today's totals from scans
    today_totals = {"protein": 0, "carbs": 0, "fats": 0}
    for scan in today_scans_data:
        for nutrient_name, nutrient_data in scan["nutrients"].items():
            if nutrient_name in ["protein"]:
                today_totals["protein"] += nutrient_data["value"]
            elif nutrient_name in ["carbohydrates", "carbs"]:
                today_totals["carbs"] += nutrient_data["value"]
            elif nutrient_name in ["fat", "fats"]:
                today_totals["fats"] += nutrient_data["value"]

    # Calculate percentages
    percent = lambda val, ref: round((val / ref) * 100) if ref else 0
    percentages = {
        'protein': percent(today_totals['protein'], recommendation['protein_g']),
        'carbs': percent(today_totals['carbs'], recommendation['carbs_g']),
        'fats': percent(today_totals['fats'], recommendation['fats_g'])
    }

    # Get scan statistics
    scan_stats = scan_storage.get_user_scan_stats(username)

    return render_template(
        'profile.html', 
        profile=profile, 
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
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('landing'))

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

@app.route('/download-report')
def download_report():
    # Example dummy response
    return "Download functionality not implemented yet."

@app.route('/edit-profile')
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    profile = load_user_health_profile(session['username'])
    if not profile:
        flash("No profile found. Please create one.")
        return redirect(url_for('profile_form'))

    return render_template('edit_profile.html', profile=profile)
# --- NEW AI ANALYSIS ROUTE ---
@app.route('/comprehensive_analysis')
def comprehensive_analysis():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    username = session['username']
    
    profile = load_user_health_profile(username)
    if not profile:
        flash("Please create a profile to get a comprehensive analysis.", "warning")
        return redirect(url_for('profile_form'))
        
    habit_summary = get_user_habit_summary(username)
    if not habit_summary or habit_summary['total_scans'] < 10:
        flash("Track at least 10 items over time to generate a comprehensive analysis.", "info")
        return redirect(url_for('profile'))
        
    ai_analysis_html = get_comprehensive_ai_analysis(profile, habit_summary)
    
    return render_template(
        'comprehensive_analysis.html', 
        ai_analysis_html=ai_analysis_html,
        user_name=profile.get('full_name', username)
    )
# Final app launch
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)