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
from io import BytesIO
import sqlite3

load_dotenv()

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
import sqlite3
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
        return redirect(url_for('dashboard'))

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
    return render_template('dashboard.html', username=session['username'])

@app.route('/scan-label', methods=['GET', 'POST'])
def scan_label():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if request.method == 'POST':
        if 'label_image' not in request.files:
            return render_template('scan_label.html', error="No file part")

        file = request.files['label_image']

        if file.filename == '':
            return render_template('scan_label.html', error="No selected file")

        if not allowed_file(file.filename):
            return render_template('scan_label.html', error="Only .jpg, .jpeg, .png files allowed")

        if not file.mimetype.startswith('image/'):
            return render_template('scan_label.html', error="Uploaded file is not an image")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        filepath = filepath.replace("\\", "/")  # Windows fix

        try:
            # Read the uploaded file once into memory
            image_stream = BytesIO(file.read())

            # Verify the image
            image_stream.seek(0)
            img = Image.open(image_stream)
            img.verify()

            # Reopen and convert the image to RGB
            image_stream.seek(0)
            img = Image.open(image_stream).convert("RGB")

            # Save the image
            img.save(filepath)

            print(f"✅ File saved to {filepath}")

            # Process the image with OCR
            raw_text, structured_nutrients = process_label_image(filepath)
            print("✅ OCR Text:", raw_text)
            print("✅ Nutrients:", structured_nutrients)

            session['scan_raw_text'] = raw_text
            session['scan_structured_nutrients'] = structured_nutrients
            return redirect(url_for('scan_result'))

        except Exception as e:
            print(f" Error while handling image: {e}")
            return render_template('scan_label.html', error=f"Could not process image: {e}")

    return render_template('scan_label.html')

@app.route('/scan-result')
def scan_result():
    raw_text = session.get('scan_raw_text')
    structured_nutrients = session.get('scan_structured_nutrients')

    if not raw_text or not structured_nutrients:
        return redirect(url_for('scan_label'))

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
        
        # Get status evaluation from database
        status_info = evaluate_nutrient_status(nutrient_name, value)
        
        # Update nutrient info
        nutrient['value'] = value  # Clean numeric value
        nutrient['status'] = status_info['status']
        nutrient['message'] = status_info['message']
        
        # Debug output
        print(f"🔍 Nutrient: {nutrient_name}, Value: {value}, Status: {status_info['status']}")

    return render_template('scan_result.html', raw_text=raw_text, structured_nutrients=structured_nutrients)


def evaluate_nutrient_status(nutrient_name, value):
    """
    Enhanced function to evaluate nutrient status with better name matching
    """
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
            'sugar': ['sugar', 'added sugar', 'total sugar'],
            'fat': ['fat', 'total fat', 'fats'],
            'saturated fat': ['saturated fat', 'sat fat'],
            'trans fat': ['trans fat'],
            'sodium': ['sodium', 'salt'],
            'carbohydrates': ['carbohydrates', 'carbs', 'total carbohydrates'],
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
        print(f"⚠️ No threshold data found for: {nutrient_name}")
        return {"status": "Unknown", "message": f"No threshold data available for {nutrient_name}."}

    max_threshold, min_threshold, high_msg, low_msg = result

    # Evaluate status based on thresholds
    if max_threshold is not None and value > max_threshold:
        return {"status": "High", "message": high_msg}
    elif min_threshold is not None and value < min_threshold:
        return {"status": "Low", "message": low_msg}
    else:
        return {"status": "Normal", "message": "This level is within the healthy range."}


# Also add this debug route to check your database content
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

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    profile = load_user_health_profile(session['username'])
    if not profile:
        flash("Your Profile Doesnt exists , Create a profile")
        return redirect(url_for('profile_form'))

    recommendation = calculate_daily_needs(profile['weight_kg'], profile['activity_level'])
    scans = load_user_scans(session['username'])

    today = datetime.now().strftime('%Y-%m-%d')
    today_scans = [s for s in scans if s['date'] == today]

    sum_nutrients = lambda key: sum(float(s.get(key, 0)) for s in today_scans)
    intake = {
        'protein_g': round(sum_nutrients('protein'), 1),
        'carbs_g': round(sum_nutrients('carbs'), 1),
        'fats_g': round(sum_nutrients('fats'), 1)
    }

    percent = lambda val, ref: round((val / ref) * 100) if ref else 0
    percentages = {
        'protein': percent(intake['protein_g'], recommendation['protein_g']),
        'carbs': percent(intake['carbs_g'], recommendation['carbs_g']),
        'fats': percent(intake['fats_g'], recommendation['fats_g'])
    }

    return render_template('profile.html', profile=profile, intake=intake, recommendation=recommendation, percentages=percentages, scans=scans, summary=get_weekly_summary(scans))

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

    # 🟩 Add this block to fix the UndefinedError
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


# Final app launch
if __name__ == '__main__':
    app.run(debug=True)
