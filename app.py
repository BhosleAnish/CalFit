import os
import re
import json
import threading
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file
)
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_session import Session
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from PIL import Image
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv

# ── project imports ───────────────────────────────────────────────────────────
from models import UserCredentialsStorage, UserProfileStorage, ScanStorage
from services.ai_service import (
    get_comprehensive_ai_analysis,
    get_personalized_nutrient_analysis_batch,
    _match_ai_result as match_ai_result,
)
from services.community import get_community_warnings_cached
from services.ocr_utils import process_label_image
from services.process_label import process_nutrition_label
from services.nlp_analyzer import analyze_report_text
from utils.auth import login_required, prevent_cache, validate_password
from utils.nutrition import (
    calculate_daily_needs,
    evaluate_nutrient_status_enhanced,
    calculate_health_score,
    get_historical_health_scores,
)

load_dotenv()

# ── app factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_static_key_123")

app.config["SESSION_TYPE"]            = "filesystem"
app.config["SESSION_PERMANENT"]       = False
app.config["SESSION_USE_SIGNER"]      = True
app.config["SESSION_FILE_DIR"]        = "./.flask_session/"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Session(app)

# ── extensions ────────────────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

csrf = CSRFProtect(app)
csrf.exempt('google_callback')

limiter = Limiter(key_func=get_remote_address, app=app)

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

# ── storage instances ─────────────────────────────────────────────────────────
credentials_storage = UserCredentialsStorage()
profile_storage     = UserProfileStorage()
scan_storage        = ScanStorage()

# ── helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def load_user_health_profile(username):
    return profile_storage.get_profile(username)


def save_user_profile(data):
    return profile_storage.save_profile(data)


def get_today_scan_visualization():
    username = session.get("username")
    if not username or scan_storage.collection is None:
        return []

    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end   = today_start + timedelta(days=1)

        today_scans = list(scan_storage.collection.find({
            "username":  username,
            "scan_date": {"$gte": today_start, "$lt": today_end},
        }).sort("scan_date", -1))

        visualization_data = []
        for scan in today_scans:
            product_name = scan.get("product_info", {}).get("name", "Unknown Product")
            nutrients    = scan.get("nutrition_analysis", {}).get("structured_nutrients", [])

            key_nutrients = {}
            for nutrient in nutrients:
                name = nutrient.get("nutrient", "").lower()
                if name in ["protein", "carbohydrates", "carbs", "fat", "fats", "sodium", "sugar"]:
                    key_nutrients[name] = {
                        "value":  nutrient.get("value", 0),
                        "unit":   nutrient.get("unit", ""),
                        "status": nutrient.get("status", "Unknown"),
                    }

            visualization_data.append({
                "product_name":    product_name,
                "scan_time":       scan.get("scan_date"),
                "nutrients":       key_nutrients,
                "total_nutrients": len(nutrients),
            })

        return visualization_data

    except Exception as e:
        print(f"❌ Error fetching today's scans: {e}")
        return []


# ── NLP warm-up ───────────────────────────────────────────────────────────────
def _warm_up_nlp():
    print("🔥 Warming up HF NLP model in background...")
    from services.nlp_analyzer import analyze_report_text
    analyze_report_text("warmup")
    print("✅ NLP model warm and ready.")

threading.Thread(target=_warm_up_nlp, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── public ────────────────────────────────────────────────────────────────────
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


# ── auth: regular ─────────────────────────────────────────────────────────────
@limiter.limit("3 per minute", methods=["POST"])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return redirect(url_for('landing'))

    username = request.form['username']
    password = request.form['password']
    user     = credentials_storage.get_user(username)

    if not user:
        flash("Account does not exist. Please sign up first.", "error")
        return redirect(url_for('landing'))

    if user.get('auth_provider') == 'google':
        flash("This account uses Google Sign-In. Please click 'Continue with Google'.", "error")
        return redirect(url_for('landing'))

    hashed = user.get('password')
    if not hashed:
        flash("Invalid account configuration. Please contact support.", "error")
        return redirect(url_for('landing'))

    if check_password_hash(hashed, password):
        session.permanent       = False
        session['username']     = username
        session['auth_provider'] = 'regular'
        flash("Logged in successfully!", "success")
        return redirect(url_for('profile'))

    flash("Incorrect password.", "error")
    return redirect(url_for('landing'))


@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    confirm  = request.form.get('confirm_password', '')

    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for('landing'))

    if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
        flash("Username may only contain letters, numbers, underscores, hyphens and dots.", "error")
        return redirect(url_for('landing'))

    if password != confirm:
        flash("Passwords do not match. Please try again.", "error")
        return redirect(url_for('landing'))

    is_valid, errors = validate_password(password)
    if not is_valid:
        for err in errors:
            flash(err, "error")
        return redirect(url_for('landing'))

    if credentials_storage.user_exists(username):
        flash("Username already exists. Please choose a different one.", "error")
        return redirect(url_for('landing'))

    if credentials_storage.add_user(username, password):
        session.permanent        = False
        session['username']      = username
        session['auth_provider'] = 'regular'
        flash("Account created successfully! Welcome to Cal-FIT.", "success")
        return redirect(url_for('profile_form'))

    flash("Failed to create account. Please try again.", "error")
    return redirect(url_for('landing'))


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash("Logged out.", "info")
    response = redirect(url_for('landing'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private, max-age=0'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response


# ── auth: google oauth ────────────────────────────────────────────────────────
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    try:
        print("Session before token:", dict(session))
        token     = google.authorize_access_token()
        user_info = token.get('userinfo')

        if not user_info:
            flash("Failed to get user information from Google.", "error")
            return redirect(url_for('landing'))

        google_id = user_info.get('sub')
        email     = user_info.get('email')
        name      = user_info.get('name')
        picture   = user_info.get('picture')
        username  = email.split('@')[0] if email else f"google_user_{google_id}"

        existing_user = credentials_storage.get_user(username)

        if existing_user:
            if existing_user.get('auth_provider') != 'google':
                flash(
                    f"An account with username '{username}' already exists. "
                    "Please login with password.", "error"
                )
                return redirect(url_for('landing'))

            credentials_storage.update_last_login(username)
            session.permanent          = False
            session['username']        = username
            session['auth_provider']   = 'google'
            session['profile_picture'] = picture
            flash(f"Welcome back, {name or username}!", "success")
            return redirect(url_for('profile'))

        # New Google user
        credentials_storage.collection.insert_one({
            "username":        username,
            "email":           email,
            "google_id":       google_id,
            "auth_provider":   "google",
            "password":        None,
            "profile_picture": picture,
            "created_at":      datetime.utcnow(),
            "last_login":      datetime.utcnow(),
        })

        profile_storage.save_profile({
            'username':  username,
            'full_name': name or '',
            'email':     email,
        })

        session.permanent          = False
        session['username']        = username
        session['auth_provider']   = 'google'
        session['profile_picture'] = picture
        flash("Account created successfully with Google! Please complete your profile.", "success")
        return redirect(url_for('profile_form'))

    except Exception as e:
        print(f"❌ Google OAuth Error: {e}")
        flash(f"Authentication failed: {e}. Please try again.", "error")
        return redirect(url_for('landing'))


limiter.exempt(google_callback)


@app.route('/auth/logout')
@login_required
def oauth_logout():
    session.clear()
    flash("Logged out successfully.", "info")
    response = redirect(url_for('landing'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private, max-age=0'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response


# ── dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
@prevent_cache
def dashboard():
    username   = session['username']
    scan_stats = scan_storage.get_user_scan_stats(username)
    return render_template('dashboard.html', username=username, scan_stats=scan_stats)


# ── scanning ──────────────────────────────────────────────────────────────────
@app.route('/scan-label', methods=['GET', 'POST'])
@login_required
@prevent_cache
def scan_label():
    if request.method == 'POST':
        if 'label_image' in request.files and request.files['label_image'].filename:
            file      = request.files['label_image']
            scan_type = "label"
        elif 'barcode_image' in request.files and request.files['barcode_image'].filename:
            file      = request.files['barcode_image']
            scan_type = "barcode"
        else:
            return render_template('scan_label.html', error="No file selected")

        if not allowed_file(file.filename):
            return render_template('scan_label.html', error="Only .jpg, .jpeg, .png files allowed")

        filename        = secure_filename(file.filename)
        timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext       = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        filepath        = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename).replace("\\", "/")

        try:
            image_stream = BytesIO(file.read())
            img          = Image.open(image_stream)

            if img.format not in ("PNG", "JPEG"):
                return render_template(
                    'scan_label.html',
                    error=f"Unsupported format: {img.format}. Please upload JPG or PNG."
                )

            img.verify()
            image_stream.seek(0)
            img = Image.open(image_stream).convert("RGB")
            img.save(filepath)

            if scan_type == "label":
                cleaned_text, structured_nutrients = process_label_image(filepath)
                session['scan_type']               = "label"
                session['scan_raw_text']           = cleaned_text
                session['scan_structured_nutrients'] = structured_nutrients
                session['scan_image_filename']     = unique_filename
                return redirect(url_for('scan_result'))

            else:  # barcode
                processed_data = process_nutrition_label(filepath)
                if not processed_data:
                    return redirect(url_for('product_not_found'))

                session['scan_type']           = "barcode"
                session['barcode_processed']   = True
                session['scan_data']           = processed_data
                session['scan_image_filename'] = unique_filename
                return redirect(url_for('index'))

        except Exception as e:
            print(f"❌ Error handling image: {e}")
            return render_template('scan_label.html', error=f"Could not process image: {e}")

    return render_template('scan_label.html')


@app.route('/product-not-found')
def product_not_found():
    return render_template('product_not_found.html')


@app.route('/scan-result')
@login_required
@prevent_cache
def scan_result():
    if session.get('scan_type') != "label":
        flash("No label scan data found.", "error")
        return redirect(url_for('scan_label'))

    raw_text           = session.get('scan_raw_text')
    structured_nutrients = session.get('scan_structured_nutrients')
    username           = session.get('username')

    if not raw_text or not structured_nutrients:
        return redirect(url_for('scan_label'))

    user_profile = load_user_health_profile(username)

    # Evaluate statuses locally
    for nutrient in structured_nutrients:
        try:
            value = float(re.sub(r'[^\d.]', '', str(nutrient['value'])))
        except (ValueError, TypeError):
            value = 0.0
        name              = nutrient.get('nutrient', '').strip()
        status_info       = evaluate_nutrient_status_enhanced(name, value)
        nutrient['value']       = value
        nutrient['status']      = status_info['status']
        nutrient['message']     = status_info['message']
        nutrient['ai_analysis'] = "Generating..."

    # Fire AI batch + community warnings concurrently
    def run_ai():
        return get_personalized_nutrient_analysis_batch(structured_nutrients, user_profile) \
            if user_profile else {}

    def run_community():
        return get_community_warnings_cached("Scanned from Label (OCR)", scan_storage)

    with ThreadPoolExecutor(max_workers=2) as ex:
        ai_future        = ex.submit(run_ai)
        community_future = ex.submit(run_community)
        ai_results       = ai_future.result()
        community_data   = community_future.result()

    # Apply AI results with fuzzy matching
    no_profile_msg = "Complete your profile to get personalized insights."
    for nutrient in structured_nutrients:
        key     = nutrient.get('nutrient', '').strip().lower()
        matched = match_ai_result(ai_results, key)
        nutrient['ai_analysis'] = matched if matched else (
            no_profile_msg if not user_profile else "No specific insight available."
        )

    # Persist scan
    try:
        scan_id = scan_storage.save_scan_data(
            username=username,
            raw_text=raw_text,
            cleaned_text=raw_text,
            structured_nutrients=structured_nutrients,
            image_filename=session.get('scan_image_filename'),
            product_name="Scanned from Label (OCR)",
            product_image_url=None,
        )
        if scan_id:
            session['last_scan_id'] = scan_id
    except Exception as e:
        print(f"❌ Error saving scan: {e}")

    return render_template(
        'scan_result.html',
        raw_text=raw_text,
        structured_nutrients=structured_nutrients,
        community_warnings=community_data,
    )


@app.route('/index')
@login_required
@prevent_cache
def index():
    barcode_processed = session.get('barcode_processed', False)
    scan_data         = session.get('scan_data', {})
    username          = session.get('username')
    community_warnings = {"total_reports": 0, "top_reports": [], "product_name": None}

    if barcode_processed and scan_data and scan_data.get('structured_nutrients') and username:
        user_profile  = load_user_health_profile(username)
        product_name  = scan_data.get('product_name')
        needs_analysis = [n for n in scan_data['structured_nutrients'] if 'ai_analysis' not in n]

        def run_ai():
            return get_personalized_nutrient_analysis_batch(needs_analysis, user_profile) \
                if user_profile and needs_analysis else {}

        def run_community():
            return get_community_warnings_cached(product_name, scan_storage) \
                if product_name else {"total_reports": 0, "top_reports": [], "product_name": None}

        with ThreadPoolExecutor(max_workers=2) as ex:
            ai_future        = ex.submit(run_ai)
            community_future = ex.submit(run_community)
            ai_results       = ai_future.result()
            community_warnings = community_future.result()

        for nutrient in scan_data['structured_nutrients']:
            if 'ai_analysis' not in nutrient:
                key = nutrient.get('nutrient', '').strip().lower()
                matched = match_ai_result(ai_results, key)
                nutrient['ai_analysis'] = matched or "No specific insight available."

    return render_template(
        'index.html',
        barcode_processed=barcode_processed,
        scan_data=scan_data,
        community_warnings=community_warnings,
    )


@app.route('/add_to_diet', methods=['POST'])
@login_required
def add_to_diet():
    username        = session.get('username')
    scan_data       = session.get('scan_data', {})
    image_filename  = session.get('scan_image_filename')
    product_name    = scan_data.get('product_name')
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
            product_image_url=product_image_url,
        )
        if scan_id:
            session.pop('scan_data', None)
            session.pop('barcode_processed', None)
            return jsonify({"status": "success", "message": "Item added to your diet successfully!"})
        return jsonify({"status": "warning", "message": "Could not add item to your diet."})

    except Exception as e:
        print(f"❌ Error saving barcode scan: {e}")
        return jsonify({"status": "error", "message": "Error saving item to your diet."})


# ── scans ─────────────────────────────────────────────────────────────────────
@app.route('/my-scans')
@login_required
@prevent_cache
def my_scans():
    username = session['username']
    scans    = scan_storage.get_user_scans(username)
    stats    = scan_storage.get_user_scan_stats(username)
    return render_template('my_scans.html', scans=scans, stats=stats)


@app.route('/view-scan/<scan_id>')
@login_required
@prevent_cache
def view_scan(scan_id):
    scan = scan_storage.get_scan_by_id(scan_id)
    if not scan or scan['username'] != session['username']:
        flash("Scan not found or access denied.", "error")
        return redirect(url_for('my_scans'))
    return render_template('view_scan.html', scan=scan)


@app.route('/delete-scan/<scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    if scan_storage.delete_scan(scan_id, session['username']):
        flash("Scan deleted successfully.", "success")
    else:
        flash("Failed to delete scan.", "error")
    return redirect(url_for('my_scans'))


@app.route('/view-all-scans')
@login_required
@prevent_cache
def view_all_scans():
    return render_template('view_all_scans.html')


# ── scans API ─────────────────────────────────────────────────────────────────
@app.route('/api/get-all-scans')
@login_required
def api_get_all_scans():
    username = session['username']

    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    try:
        scans = list(scan_storage.collection.find({"username": username}).sort("scan_date", -1))

        for scan in scans:
            scan['_id'] = str(scan['_id'])
            if isinstance(scan.get('scan_date'), datetime):
                scan['scan_date'] = scan['scan_date'].isoformat()

        week_ago     = datetime.utcnow() - timedelta(days=7)
        recent_scans = sum(
            1 for s in scans
            if 'scan_date' in s and datetime.fromisoformat(s['scan_date']) >= week_ago
        )
        high_risk_count = sum(
            s.get('nutrition_analysis', {}).get('summary', {}).get('high_risk_count', 0)
            for s in scans
        )

        return jsonify({
            "success": True,
            "scans":   scans,
            "stats": {
                "total_scans":    len(scans),
                "recent_scans":   recent_scans,
                "high_risk_count": high_risk_count,
            },
        })

    except Exception as e:
        print(f"❌ Error fetching scans: {e}")
        return jsonify({"success": False, "error": "Failed to fetch scans"}), 500


@app.route('/api/delete-scan/<scan_id>', methods=['DELETE'])
@login_required
def api_delete_scan(scan_id):
    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    try:
        obj_id = ObjectId(scan_id)
    except (InvalidId, TypeError):
        return jsonify({"success": False, "error": "Invalid scan ID format"}), 400

    try:
        result = scan_storage.collection.delete_one({"_id": obj_id, "username": session['username']})
        if result.deleted_count > 0:
            return jsonify({"success": True, "message": "Scan deleted successfully"})
        return jsonify({"success": False, "error": "Scan not found or access denied"}), 404

    except Exception as e:
        print(f"❌ Error deleting scan: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route('/api/submit-report/<scan_id>', methods=['POST'])
@login_required
def api_submit_report(scan_id):
    if scan_storage.collection is None:
        return jsonify({"success": False, "error": "Database not available"}), 503

    try:
        obj_id = ObjectId(scan_id)
    except (InvalidId, TypeError):
        return jsonify({"success": False, "error": "Invalid scan ID format"}), 400

    report_data = request.get_json()
    if not report_data or not report_data.get('description', '').strip():
        return jsonify({"success": False, "error": "Report description is required"}), 400

    payload = {
        "reported_at":     datetime.utcnow(),
        "description":     report_data['description'].strip(),
        "severity":        report_data.get('severity'),
        "contact_consent": report_data.get('contact_consent', False),
        "status":          "submitted",
    }

    try:
        result = scan_storage.collection.update_one(
            {"_id": obj_id, "username": session['username']},
            {"$set": {"user_feedback.issue_report": payload}},
        )
        if result.matched_count > 0:
            return jsonify({"success": True, "message": "Report submitted successfully"})
        return jsonify({"success": False, "error": "Scan not found or access denied"}), 404

    except Exception as e:
        print(f"❌ Error submitting report: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


# ── profile ───────────────────────────────────────────────────────────────────
@app.route('/profile')
@login_required
@prevent_cache
def profile():
    username     = session['username']
    profile_data = load_user_health_profile(username)

    if not profile_data:
        flash("Your profile doesn't exist. Please create one.", "warning")
        return redirect(url_for('profile_form'))

    try:
        recommendation = calculate_daily_needs(
            profile_data['weight_kg'], profile_data['activity_level']
        )
    except (ValueError, KeyError, TypeError):
        flash("Your profile is incomplete. Please update your weight and activity level.", "warning")
        return redirect(url_for('edit_profile'))

    today_scans_data = get_today_scan_visualization()

    today_totals = {"protein": 0, "carbs": 0, "fats": 0}
    for scan in today_scans_data:
        for name, data in scan["nutrients"].items():
            if name == "protein":
                today_totals["protein"] += data["value"]
            elif name in ["carbohydrates", "carbs"]:
                today_totals["carbs"]   += data["value"]
            elif name in ["fat", "fats"]:
                today_totals["fats"]    += data["value"]

    def pct(val, ref):
        return round((val / ref) * 100) if ref else 0

    percentages = {
        'protein': pct(today_totals['protein'], recommendation['protein_g']),
        'carbs':   pct(today_totals['carbs'],   recommendation['carbs_g']),
        'fats':    pct(today_totals['fats'],    recommendation['fats_g']),
    }

    scan_stats = scan_storage.get_user_scan_stats(username)

    return render_template(
        'profile.html',
        profile=profile_data,
        intake={
            'protein_g': round(today_totals['protein'], 1),
            'carbs_g':   round(today_totals['carbs'],   1),
            'fats_g':    round(today_totals['fats'],    1),
        },
        recommendation=recommendation,
        percentages=percentages,
        today_scans=today_scans_data,
        scan_stats=scan_stats,
    )


@app.route('/profile-form', methods=['GET', 'POST'])
@login_required
@prevent_cache
def profile_form():
    if request.method == 'POST':
        profile_data = {
            'username':          session['username'],
            'full_name':         request.form.get('full_name', ''),
            'height_cm':         request.form.get('height_cm', ''),
            'weight_kg':         request.form.get('weight_kg', ''),
            'age':               request.form.get('age', ''),
            'gender':            request.form.get('gender', ''),
            'activity_level':    request.form.get('activity_level', ''),
            'medical_conditions': request.form.get('medical_conditions', ''),
            'allergies':         request.form.get('allergies', ''),
        }
        if save_user_profile(profile_data):
            flash("Profile saved successfully!", "success")
            return redirect(url_for('profile'))
        flash("Failed to save profile. Please try again.", "error")

    return render_template('profile_form.html')


@app.route('/edit-profile', methods=['GET', 'POST'])
@login_required
@prevent_cache
def edit_profile():
    username = session['username']

    if request.method == 'POST':
        profile_data = {
            'username':          username,
            'full_name':         request.form.get('full_name', ''),
            'height_cm':         request.form.get('height_cm', ''),
            'weight_kg':         request.form.get('weight_kg', ''),
            'age':               request.form.get('age', ''),
            'gender':            request.form.get('gender', ''),
            'activity_level':    request.form.get('activity_level', ''),
            'medical_conditions': request.form.get('medical_conditions', ''),
            'allergies':         request.form.get('allergies', ''),
        }
        if save_user_profile(profile_data):
            flash("Profile updated successfully!", "success")
            return redirect(url_for('profile'))
        flash("Failed to update profile. Please try again.", "error")

    profile = load_user_health_profile(username)
    if not profile:
        flash("No profile found. Please create one.", "warning")
        return redirect(url_for('profile_form'))

    return render_template('edit_profile.html', profile=profile)


@app.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    username = session['username']
    try:
        credentials_storage.delete_user(username)
        profile_storage.delete_profile(username)
        scan_storage.delete_user_scans(username)
        session.clear()
        return jsonify({"success": True, "message": "Account deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── AI analysis ───────────────────────────────────────────────────────────────
@app.route('/get-ai-analysis', methods=['POST'])
@login_required
def get_ai_analysis():
    try:
        analysis_html = get_comprehensive_ai_analysis(
            username=session['username'],
            scan_storage=scan_storage,
            profile_storage=profile_storage,
        )
        return jsonify({"success": True, "analysis": analysis_html})
    except Exception as e:
        print(f"❌ Error generating AI analysis: {e}")
        return jsonify({"error": "Failed to generate analysis"}), 500


# ── charts / data ─────────────────────────────────────────────────────────────
@app.route('/get-health-score-data/<period>')
@login_required
def get_health_score_data(period='weekly'):
    username = session['username']
    profile  = load_user_health_profile(username)

    if not profile:
        return jsonify({"error": "Profile not found"}), 404

    if not all(profile.get(k) for k in ['weight_kg', 'activity_level']):
        return jsonify({"error": "Profile incomplete"}), 400

    if period not in ('daily', 'weekly', 'monthly'):
        return jsonify({"error": "Invalid period"}), 400

    try:
        recommendations = calculate_daily_needs(profile['weight_kg'], profile['activity_level'])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Could not calculate recommendations: {e}"}), 400

    data = get_historical_health_scores(
        username, period, recommendations, scan_storage.collection
    )

    return jsonify({
        "labels":       data["labels"],
        "actual_scores": data["scores"],
        "goal_scores":  [85] * len(data["labels"]),
    })


@app.route('/get-scan-count-data/<period>')
@login_required
def get_scan_count_data(period='weekly'):
    username = session['username']

    if scan_storage.collection is None:
        return jsonify({"error": "Database not available"}), 503

    if period not in ('daily', 'weekly', 'monthly'):
        return jsonify({"error": "Invalid period"}), 400

    end_date = datetime.utcnow()

    if period == 'daily':
        start_date  = end_date - timedelta(days=30)
        group_id    = {"$dateToString": {"format": "%Y-%m-%d", "date": "$scan_date"}}
    elif period == 'weekly':
        start_date  = end_date - timedelta(weeks=12)
        group_id    = {"$dateToString": {"format": "%Y-W%U", "date": "$scan_date"}}
    else:
        start_date  = end_date - timedelta(days=365)
        group_id    = {"$dateToString": {"format": "%Y-%m", "date": "$scan_date"}}

    try:
        results = list(scan_storage.collection.aggregate([
            {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
            {"$group": {"_id": group_id, "scan_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]))

        labels      = [r['_id'] for r in results]
        scan_counts = [r['scan_count'] for r in results]

        formatted = []
        for label in labels:
            if period == 'weekly':
                try:
                    year, week = label.split('-W')
                    formatted.append(f"Week {week}, {year}")
                except Exception:
                    formatted.append(label)
            elif period == 'monthly':
                try:
                    year, month = label.split('-')
                    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                                   'Jul','Aug','Sep','Oct','Nov','Dec']
                    formatted.append(f"{month_names[int(month)-1]} {year}")
                except Exception:
                    formatted.append(label)
            else:
                formatted.append(label)

        return jsonify({
            "labels":      formatted,
            "scan_counts": scan_counts,
            "total_scans": sum(scan_counts),
        })

    except Exception as e:
        print(f"❌ Error fetching scan count data: {e}")
        return jsonify({"error": "Failed to fetch scan data"}), 500


# ── food tracker ──────────────────────────────────────────────────────────────
@app.route('/food-tracker', methods=['GET', 'POST'])
@login_required
@prevent_cache
def food_tracker():
    if 'food_items' not in session:
        session['food_items'] = []

    if request.method == 'POST':
        if 'food_image' in request.files and request.files['food_image'].filename:
            image    = request.files['food_image']
            nutrients = process_label_image(image)
            if nutrients:
                session['food_items'].append({
                    'item':     'Scanned Item',
                    'quantity': 1,
                    'protein':  nutrients.get('protein', 0),
                    'carbs':    nutrients.get('carbohydrates', 0),
                    'fats':     nutrients.get('fats', 0),
                })
                session.modified = True
                flash("Scanned food item added.")
        else:
            item     = request.form.get('item')
            quantity = float(request.form.get('quantity', 0))
            protein  = float(request.form.get('protein', 0))
            carbs    = float(request.form.get('carbs', 0))
            fats     = float(request.form.get('fats', 0))

            updated = False
            for food in session['food_items']:
                if food['item'].lower() == item.lower():
                    food['quantity'] += quantity
                    food['protein']  += protein
                    food['carbs']    += carbs
                    food['fats']     += fats
                    updated = True
                    break

            if not updated:
                session['food_items'].append({
                    'item': item, 'quantity': quantity,
                    'protein': protein, 'carbs': carbs, 'fats': fats,
                })
            session.modified = True
            flash("Food item added/updated.")

    total = {
        'protein': sum(i['protein'] for i in session['food_items']),
        'carbs':   sum(i['carbs']   for i in session['food_items']),
        'fats':    sum(i['fats']    for i in session['food_items']),
    }

    return render_template('food_tracker.html', food_items=session['food_items'], total=total)


# ── misc / debug ──────────────────────────────────────────────────────────────
@app.route('/download-report')
@login_required
def download_report():
    return "Download functionality not implemented yet."


@app.route('/admin/cleanup-scans')
@login_required
def cleanup_scans():
    deleted = scan_storage.cleanup_expired_scans()
    return jsonify({"message": f"Cleaned up {deleted} expired scans"})


@app.route('/debug/mongodb-collections')
@login_required
@prevent_cache
def mongodb_collections():
    try:
        db = credentials_storage.mongo_config.db
        html = f"""
        <h2>MongoDB Collections Status</h2>
        <div style="font-family:monospace;background:#f5f5f5;padding:20px;border-radius:5px">
            <h3>Database: {db.name}</h3>
            <p><strong>User Credentials:</strong> {credentials_storage.collection.count_documents({})} users</p>
            <p><strong>User Profiles:</strong>    {profile_storage.collection.count_documents({})} profiles</p>
            <p><strong>Scan Data:</strong>         {scan_storage.collection.count_documents({})} scans</p>
            <hr>
            <p><strong>Collections:</strong></p>
            <ul>{''.join(f'<li>{c}</li>' for c in db.list_collection_names())}</ul>
        </div>
        <p><a href="{url_for('dashboard')}">← Back to Dashboard</a></p>
        """
        return html
    except Exception as e:
        return f"<h2>MongoDB Error</h2><p>{e}</p>"


@app.route('/debug/my-profile-info')
@login_required
@prevent_cache
def my_profile_info():
    username = session['username']
    profile  = load_user_health_profile(username)
    if not profile:
        return "<h2>No Profile Found</h2><p>Please create your profile first.</p>"

    html = f"""
    <h2>Your Profile Information</h2>
    <div style="font-family:monospace;background:#f5f5f5;padding:20px;border-radius:5px">
        <p><strong>Username:</strong>          {profile.get('username')}</p>
        <p><strong>Full Name:</strong>         {profile.get('full_name')}</p>
        <p><strong>Age:</strong>               {profile.get('age')}</p>
        <p><strong>Gender:</strong>            {profile.get('gender')}</p>
        <p><strong>Height:</strong>            {profile.get('height_cm')} cm</p>
        <p><strong>Weight:</strong>            {profile.get('weight_kg')} kg</p>
        <p><strong>Activity Level:</strong>    {profile.get('activity_level')}</p>
        <p><strong>Medical Conditions:</strong>{profile.get('medical_conditions')}</p>
        <p><strong>Allergies:</strong>         {profile.get('allergies')}</p>
        <p><strong>Created:</strong>           {profile.get('created_at')}</p>
        <p><strong>Updated:</strong>           {profile.get('updated_at')}</p>
    </div>
    <p><a href="{url_for('profile')}">← Back to Profile</a></p>
    """
    return html


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="localhost", debug=True, use_reloader=False)