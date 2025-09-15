import pytesseract
import re
import sqlite3
from PIL import Image
import requests
import zxingcpp  # <-- NEW LIBRARY IMPORT

# ==============================================================================
# SECTION 1: API-BASED EXTRACTION (Using the new zxing-cpp library)
# ==============================================================================

def get_barcode_from_image(image_path):
    """Scans an image using the zxing-cpp library and returns the first barcode found."""
    ### DEBUG ###
    print("\n[DEBUG] Step 1: Attempting to find barcode with zxing-cpp...")
    try:
        # Open the image file
        image = Image.open(image_path)
        # Read all barcodes from the image
        results = zxingcpp.read_barcodes(image)
        
        if results:
            # Get the text from the first barcode found
            data = results[0].text
            ### DEBUG ###
            print(f"[DEBUG] Barcode found by zxing-cpp. Data: '{data}'")
            return data
    except Exception as e:
        print(f"[DEBUG] An error occurred during zxing-cpp barcode detection: {e}")

    ### DEBUG ###
    print("[DEBUG] No barcode was found by zxing-cpp.")
    return None

# --- [The rest of the file remains the same] ---

def get_product_from_api(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    ### DEBUG ###
    print(f"[DEBUG] Querying API: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == 1 and "product" in data:
            ### DEBUG ###
            print("[DEBUG] Product found in API.")
            return data["product"]
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] API request failed: {e}")
    ### DEBUG ###
    print("[DEBUG] Product not found in API for the given barcode.")
    return None

def parse_api_nutrients(product_data, thresholds_db):
    nutrients = {}
    api_nutrients = product_data.get('nutriments', {})

    for our_name, off_key in OPENFOODFACTS_MAPPING.items():
        if off_key not in api_nutrients:
            continue

        value = api_nutrients.get(off_key)
        if value is None:
            continue

        # Units in OpenFoodFacts use <nutrient>_unit
        unit_key = off_key.replace("_100g", "_unit")
        unit = api_nutrients.get(unit_key, 'g')

        try:
            nutrients[our_name] = {"value": float(value), "unit": unit}
        except (TypeError, ValueError):
            continue

    return combine_nutrients_with_thresholds(nutrients, thresholds_db)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

NUTRIENT_SPECS = {
    "total fat":        {"aliases": ["fat", "total fats"], "unit": "g", "category": "macronutrient"},
    "saturated fat":    {"aliases": ["sat fat", "saturated fats"], "unit": "g", "category": "macronutrient"},
    "trans fat":        {"aliases": ["trans fats"], "unit": "g", "category": "macronutrient"},
    "cholesterol":      {"aliases": [], "unit": "mg", "category": "mineral"},
    "sodium":           {"aliases": ["salt"], "unit": "mg", "category": "mineral"},
    "total carbohydrate": {"aliases": ["total carb", "total carbs", "carbs", "carbohydrate", "carbohydrates", "total carb."], "unit": "g", "category": "macronutrient"},
    "dietary fiber":    {"aliases": ["fiber"], "unit": "g", "category": "macronutrient"},
    "total sugars":     {"aliases": ["total sugar", "sugars"], "unit": "g", "category": "macronutrient"},
    "added sugar":      {"aliases": ["added sugars", "incl. added sugars", "includes added sugars", "incl added sugars"], "unit": "g", "category": "macronutrient"},
    "protein":          {"aliases": [], "unit": "g", "category": "macronutrient"},
    "vitamin d":        {"aliases": ["vit d", "vitamin d."], "unit": "mcg", "category": "vitamin"},
    "calcium":          {"aliases": [], "unit": "mg", "category": "mineral"},
    "iron":             {"aliases": [], "unit": "mg", "category": "mineral"},
    "potassium":        {"aliases": [], "unit": "mg", "category": "mineral"},
}
# --- Add this mapping near the top, after NUTRIENT_SPECS ---
OPENFOODFACTS_MAPPING = {
    "calories": "energy-kcal_100g",
    "protein": "proteins_100g",
    "total fat": "fat_100g",
    "saturated fat": "saturated-fat_100g",
    "trans fat": "trans-fat_100g",
    "cholesterol": "cholesterol_100g",
    "sodium": "sodium_100g",
    "total carbohydrate": "carbohydrates_100g",
    "dietary fiber": "fiber_100g",
    "total sugars": "sugars_100g",
    "added sugar": "added-sugars_100g",
    "vitamin d": "vitamin-d_100g",
    "calcium": "calcium_100g",
    "iron": "iron_100g",
    "potassium": "potassium_100g"
}

def _basic_ocr_fixes(s: str) -> str:
    s = s.replace("Âµg", "mcg")
    s = re.sub(r'\b[lI]ron\b', 'iron', s, flags=re.IGNORECASE)
    s = re.sub(r'\bimg\b', 'mg', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOg\b', '0g', s, flags=re.IGNORECASE)
    s = re.sub(r'\b0 g\b', '0g', s, flags=re.IGNORECASE)
    s = re.sub(r'«', '', s)
    s = re.sub(r'\b59\s+6%\b', '5g 6%', s, flags=re.IGNORECASE)
    s = re.sub(r'\b129\s+43%\b', '12g 43%', s, flags=re.IGNORECASE)
    return s

def clean_ocr_text(raw_text: str) -> str:
    raw_text = _basic_ocr_fixes(raw_text)
    lines = [ln.strip() for ln in raw_text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln or len(ln) < 2: continue
        if re.search(r'nutrition facts|serving size|per serving|per container|daily value|calories a day', ln, re.I): continue
        if not any(c.isalpha() for c in ln) and not re.search(r'\d', ln): continue
        ln = re.sub(r'\s+', ' ', ln).strip()
        if ln.endswith('.'): ln = ln[:-1]
        cleaned.append(ln)
    return "\n".join(cleaned)

def find_nutrient_in_line(line):
    line_lower = line.lower()
    patterns = [
        (r'\btotal\s+fat\b', 'total fat'), (r'\bsaturated\s+fat\b', 'saturated fat'),
        (r'\btrans\s+fat\b', 'trans fat'), (r'\bcholesterol\b', 'cholesterol'),
        (r'\bsodium\b', 'sodium'), (r'\btotal\s+carb\w*\b', 'total carbohydrate'),
        (r'\bdietary\s+fiber\b', 'dietary fiber'), (r'\btotal\s+sugars\b', 'total sugars'),
        (r'\bincl\.\s*added\s+sugars\b', 'added sugar'), (r'\badded\s+sugars\b', 'added sugar'),
        (r'\bprotein\b', 'protein'), (r'\bvitamin\s+d\b', 'vitamin d'),
        (r'\bcalcium\b', 'calcium'), (r'\biron\b', 'iron'), (r'\bpotassium\b', 'potassium'),
    ]
    for pattern, canonical in patterns:
        match = re.search(pattern, line_lower)
        if match: return canonical, match.start()
    return None, -1

def extract_values_from_line(line):
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(mcg|mg|g)\b', r'(\d+(?:\.\d+)?)(g|mg|mcg)\s+\d+%',
        r'(\d+)\s+(g|mg|mcg)', r'(\d+(?:\.\d+)?)(g|mg|mcg)',
    ]
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, line, re.IGNORECASE)
        for match in matches:
            try:
                values.append((float(match[0]), match[1].lower()))
            except (ValueError, IndexError):
                continue
    seen, unique_values = set(), []
    for v, u in values:
        if (v, u) not in seen:
            seen.add((v, u)); unique_values.append((v, u))
    return unique_values

def extract_nutrients(ocr_text: str):
    lines = ocr_text.splitlines()
    results = {}
    for i, line in enumerate(lines):
        if not line.strip(): continue
        nutrient_name, pos = find_nutrient_in_line(line)
        if not nutrient_name: continue
        spec = NUTRIENT_SPECS.get(nutrient_name, {"unit": "g"})
        expected_unit = spec.get("unit", "g")
        found_value = None
        search_lines = [line]
        for j in range(i + 1, min(i + 3, len(lines))):
            next_line = lines[j].strip()
            if not next_line or find_nutrient_in_line(next_line)[0]: break
            search_lines.append(next_line)
        all_values = []
        for search_line in search_lines:
            all_values.extend(extract_values_from_line(search_line))
        if all_values:
            for value, unit in all_values:
                if unit == expected_unit.lower():
                    found_value = (value, unit); break
            if not found_value:
                found_value = all_values[0]
        if found_value:
            value, unit = found_value
            results[nutrient_name] = {"value": value, "unit": unit}
    return results

def process_image_with_ocr(image_path, thresholds_db):
    ### DEBUG ###
    print("[DEBUG] Step 2: Falling back to OCR method.")
    try:
        raw_text = pytesseract.image_to_string(Image.open(image_path))
        ### DEBUG ###
        print(f"[DEBUG] Raw OCR text (first 300 chars): '{raw_text[:300].strip()}...'")
        
        cleaned_text = clean_ocr_text(raw_text)
        ### DEBUG ###
        print(f"[DEBUG] Cleaned OCR text: '{cleaned_text}'")

        parsed_nutrients = extract_nutrients(cleaned_text)
        ### DEBUG ###
        print(f"[DEBUG] Nutrients parsed from OCR: {parsed_nutrients}")
        
        if not parsed_nutrients:
            ### DEBUG ###
            print("[DEBUG] OCR parsing resulted in NO nutrients found.")
            return None

        return combine_nutrients_with_thresholds(parsed_nutrients, thresholds_db)
    except Exception as e:
        print(f"[DEBUG] An error occurred during OCR processing: {e}")
        return None

def load_thresholds_from_db():
    try:
        conn = sqlite3.connect('food_thresholds.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, unit, max_threshold, min_threshold, category FROM ingredient_thresholds")
        rows = cursor.fetchall()
        conn.close()
        thresholds = {}
        for name, unit, max_val, min_val, category in rows:
            thresholds[name.lower()] = {
                "unit": (unit or '').lower(), "max_threshold": max_val,
                "min_threshold": min_val, "category": category or "unknown"
            }
        return thresholds
    except sqlite3.Error as e:
        print(f"Database error: {e}. Make sure food_thresholds.db exists and is correct.")
        return {}

def convert_unit(value, from_unit, to_unit):
    from_unit, to_unit = (from_unit or '').lower(), (to_unit or '').lower()
    if from_unit == to_unit or not from_unit or not to_unit: return value, to_unit
    factors = {
        ("g", "mg"): 1000.0, ("mg", "g"): 0.001, ("mcg", "mg"): 0.001,
        ("mg", "mcg"): 1000.0, ("g", "mcg"): 1_000_000.0, ("mcg", "g"): 1e-6,
    }
    return (value * factors[(from_unit, to_unit)], to_unit) if (from_unit, to_unit) in factors else (None, from_unit)

def combine_nutrients_with_thresholds(nutrients, thresholds_db):
    combined = []
    for name, info in nutrients.items():
        value, unit = info["value"], info["unit"].lower()
        threshold = thresholds_db.get(name.lower())
        category, status = "unknown", "Unknown"
        if threshold:
            desired_unit = threshold.get("unit") or unit
            converted, out_unit = convert_unit(value, unit, desired_unit)
            if converted is not None: value, unit = converted, out_unit
            category = threshold.get("category", "unknown")
            max_val, min_val = threshold.get("max_threshold"), threshold.get("min_threshold")
            status = "Normal"
            if max_val is not None and value > max_val: status = "High"
            if min_val is not None and value < min_val: status = "Low"
        elif name.lower() in NUTRIENT_SPECS:
            category = NUTRIENT_SPECS[name.lower()].get("category", "unknown")
        
        combined.append({
            "nutrient": name.title().replace(" Of ", " of "),
            "value": round(value, 2), "unit": unit,
            "status": status, "category": category
        })
        
    return combined

def process_nutrition_label(image_path):
    """Main function to process a nutrition label."""
    print("\n[DEBUG] --- Starting Nutrition Label Process ---")
    db_thresholds = load_thresholds_from_db()
    if not db_thresholds:
        print("[DEBUG] Could not load thresholds from database. Aborting.")
        return None

    # Step 1: Try Barcode/API method
    barcode = get_barcode_from_image(image_path)
    if barcode and barcode.strip():
        product_data = get_product_from_api(barcode)
        if product_data:
            print("[DEBUG] SUCCESS: Product found in Open Food Facts API.")
            
            product_name = product_data.get('product_name', 'Unknown Product')
            product_image_url = product_data.get('image_front_url', None)

            nutrient_analysis = parse_api_nutrients(product_data, db_thresholds)
            
            return {
                "product_name": product_name,
                "product_image_url": product_image_url,
                "structured_nutrients": nutrient_analysis,
                "source": "api"
            }
        else:
            print("[DEBUG] Barcode was found, but product was not in the API database. Trying OCR...")

    # Step 2: Fallback to OCR method
    ocr_results = process_image_with_ocr(image_path, db_thresholds)
    if ocr_results:
        return {
            "product_name": "Scanned from Label (OCR)",
            "product_image_url": None,
            "structured_nutrients": ocr_results,
            "source": "ocr"
        }
    
    return None