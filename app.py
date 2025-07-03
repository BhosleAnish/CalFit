from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from ocr_utils import process_label_image
import csv, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'

USER_CSV = 'users.csv'
PROFILE_CSV = 'user_profiles.csv'
SCAN_CSV = 'user_scans.csv'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CSV Initialization
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

def add_user(username, password):
    with open(USER_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([username, password])

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
    mult = {'sedentary': 25, 'moderate': 30, 'active': 35}.get(activity_level, 30)
    calories = weight_kg * mult
    return {
        "calories": round(calories),
        "protein_g": round(weight_kg * 1.2),
        "fats_g": round((0.25 * calories) / 9),
        "carbs_g": round((0.50 * calories) / 4)
    }

def load_user_scans(username):
    if not os.path.exists(SCAN_CSV): return []
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

# ------------------------- Routes -------------------------

@app.route('/')
def landing(): return render_template('landing.html')

@app.route('/about')
def about(): return render_template('about_us.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    users = load_users()
    if users.get(username) == password:
        session['username'] = username
        if not load_user_health_profile(username):
            return redirect(url_for('profile_form'))
        flash("Logged in successfully!", "success")
        return redirect(url_for('dashboard'))
    flash("Invalid credentials.", "error")
    return redirect(url_for('landing'))

@app.route('/signup', methods=['POST'])
def signup():
    username, password = request.form['username'], request.form['password']
    if username in load_users():
        flash("Username already exists.", "error")
        return redirect(url_for('landing'))
    add_user(username, password)
    session['username'] = username
    flash("Signup successful!", "success")
    return redirect(url_for('profile_form'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash("Please log in.", "error")
        return redirect(url_for('landing'))
    return render_template('dashboard.html', username=session['username'])
@app.route('/scan-label', methods=['POST'])
def scan_label():
    if 'username' not in session:
        return redirect(url_for('landing'))

    file = request.files.get('label_image')
    if not file or file.filename == '':
        flash("No file selected for scanning.", "error")
        return redirect(url_for('dashboard'))

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    ocr_result, risks = process_label_image(filepath)

    return render_template('scan_result.html', ocr_result=ocr_result, risks=risks)

@app.route('/profile-form', methods=['GET', 'POST'])
def profile_form():
    if 'username' not in session: return redirect(url_for('landing'))
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
        flash("Profile saved!", "success")
        return redirect(url_for('profile'))
    return render_template('profile_form.html')

@app.route('/profile')
def profile():
    if 'username' not in session: return redirect(url_for('landing'))
    profile = load_user_health_profile(session['username'])
    if not profile: return redirect(url_for('profile_form'))

    weight = float(profile.get('weight_kg', 0))
    activity = profile.get('activity_level', 'moderate').lower()
    recommendation = calculate_daily_needs(weight, activity)

    scans = load_user_scans(session['username'])
    today = datetime.now().strftime('%Y-%m-%d')
    today_scans = [s for s in scans if s['date'] == today]
    total = lambda key: sum(float(s.get(key, 0)) for s in today_scans)

    intake = {
        'protein_g': round(total('protein'), 1),
        'carbs_g': round(total('carbs'), 1),
        'fats_g': round(total('fats'), 1)
    }

    percent = lambda a, b: round((a / b) * 100) if b else 0
    percentages = {
        'protein': percent(intake['protein_g'], recommendation['protein_g']),
        'carbs': percent(intake['carbs_g'], recommendation['carbs_g']),
        'fats': percent(intake['fats_g'], recommendation['fats_g']),
    }

    return render_template('profile.html', profile=profile, intake=intake, recommendation=recommendation, percentages=percentages, scans=scans, summary=get_weekly_summary(scans))
@app.route('/download-report')
def download_report():
    if 'username' not in session:
        return redirect(url_for('landing'))

    username = session['username']
    user_scans = load_user_scans(username)

    # Create a temporary CSV file
    report_path = f"{username}_report.csv"
    with open(report_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['item', 'quantity', 'protein', 'carbs', 'fats', 'date'])
        writer.writeheader()
        for scan in user_scans:
            writer.writerow({
                'item': scan['item'],
                'quantity': scan['quantity'],
                'protein': scan['protein'],
                'carbs': scan['carbs'],
                'fats': scan['fats'],
                'date': scan['date']
            })

    from flask import send_file
    return send_file(report_path, as_attachment=True)
@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('landing'))

    username = session['username']
    profile = load_user_health_profile(username)

    if not profile:
        return redirect(url_for('profile_form'))

    if request.method == 'POST':
        updated_data = {
            'username': username,
            'full_name': request.form['full_name'],
            'age': request.form['age'],
            'gender': request.form['gender'],
            'height_cm': request.form['height_cm'],
            'weight_kg': request.form['weight_kg'],
            'activity_level': request.form['activity_level'],
            'medical_conditions': request.form['medical_conditions'],
            'allergies': request.form['allergies']
        }
        save_user_profile(updated_data)
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('edit_profile.html', profile=profile)

@app.route('/food-tracker', methods=['GET', 'POST'])
def food_tracker():
    if 'username' not in session:
        return redirect(url_for('landing'))

    food_items = session.get('food_items', [])

    if request.method == 'POST':
        action = request.form.get('action')



        #Scan image and extract values
        if action == 'scan':
            file = request.files.get('label_image')
            if not file or file.filename == '':
                flash("No file selected for scan.", "error")
                return redirect(url_for('food_tracker'))

            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            text, _ = process_label_image(filepath)

            def extract_nutrient(name):
                match = re.search(rf"{name}[:\-]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
                return float(match.group(1)) if match else 0

            entry = {
                'item': 'Scanned Label',
                'quantity': 100,
                'protein': extract_nutrient('protein'),
                'carbs': extract_nutrient('carbs'),
                'fats': extract_nutrient('fats')
            }
            food_items.append(entry)
            session['food_items'] = food_items
            return redirect(url_for('food_tracker'))

        # 👇 Manual Add
        else:
            item = request.form.get('item')
            quantity = request.form.get('quantity')
            if not item or not quantity:
                flash("Both fields are required.", "error")
                return redirect(url_for('food_tracker'))

            try:
                quantity = float(quantity)
            except ValueError:
                flash("Invalid quantity.", "error")
                return redirect(url_for('food_tracker'))

            food_db = {
                'banana': {'protein': 1.1, 'carbs': 23, 'fats': 0.3},
                'apple': {'protein': 0.3, 'carbs': 14, 'fats': 0.2},
                'rice': {'protein': 2.7, 'carbs': 28, 'fats': 0.3},
                'egg': {'protein': 6, 'carbs': 0.6, 'fats': 5}
            }
            data = food_db.get(item.lower(), {'protein': 0.5, 'carbs': 5, 'fats': 0.2})
            entry = {
                'item': item,
                'quantity': quantity,
                'protein': round(data['protein'] * quantity / 100, 2),
                'carbs': round(data['carbs'] * quantity / 100, 2),
                'fats': round(data['fats'] * quantity / 100, 2)
            }
            food_items.append(entry)
            session['food_items'] = food_items

    total = {
        'protein': round(sum(i['protein'] for i in food_items), 2),
        'carbs': round(sum(i['carbs'] for i in food_items), 2),
        'fats': round(sum(i['fats'] for i in food_items), 2)
    }
    return render_template('food_tracker.html', food_items=food_items, total=total)


@app.route('/delete-item/<int:index>', methods=['POST'])
def delete_food_item(index):
    if 'food_items' in session:
        items = session['food_items']
        if 0 <= index < len(items):
            items.pop(index)
            session['food_items'] = items
    return redirect(url_for('food_tracker'))

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('landing'))

# Run server
if __name__ == '__main__':
    app.run(debug=True)
