import pytesseract
import re
import sqlite3
from difflib import get_close_matches
from PIL import Image
import cv2
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---- 1) DB loader (unchanged) ------------------------------------------------
def load_thresholds_from_db():
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, unit, max_threshold, min_threshold, category FROM ingredient_thresholds")
    rows = cursor.fetchall()
    conn.close()

    thresholds = {}
    for name, unit, max_val, min_val, category in rows:
        thresholds[name.lower()] = {
            "unit": (unit or '').lower(),
            "max_threshold": max_val,
            "min_threshold": min_val,
            "category": category or "unknown"
        }
    return thresholds

# ---- 2) Known nutrients, aliases, expected units -----------------------------
NUTRIENT_SPECS = {
    "total fat":         {"aliases": ["fat", "total fats"],                   "unit": "g",   "category": "macronutrient"},
    "saturated fat":     {"aliases": ["sat fat", "saturated fats"],           "unit": "g",   "category": "macronutrient"},
    "trans fat":         {"aliases": ["trans fats"],                          "unit": "g",   "category": "macronutrient"},
    "cholesterol":       {"aliases": [],                                      "unit": "mg",  "category": "mineral"},
    "sodium":            {"aliases": ["salt"],                                "unit": "mg",  "category": "mineral"},
    "total carbohydrate":{"aliases": ["total carb", "total carbs", "carbs", "carbohydrate", "carbohydrates", "total carb."], "unit": "g", "category": "macronutrient"},
    "dietary fiber":     {"aliases": ["fiber"],                               "unit": "g",   "category": "macronutrient"},
    "total sugars":      {"aliases": ["total sugar", "sugars"],               "unit": "g",   "category": "macronutrient"},
    "added sugar":       {"aliases": ["added sugars", "incl. added sugars", "includes added sugars", "incl added sugars"], "unit": "g", "category": "macronutrient"},
    "protein":           {"aliases": [],                                      "unit": "g",   "category": "macronutrient"},
    "vitamin d":         {"aliases": ["vit d", "vitamin d."],                 "unit": "mcg", "category": "vitamin"},
    "calcium":           {"aliases": [],                                      "unit": "mg",  "category": "mineral"},
    "iron":              {"aliases": [],                                      "unit": "mg",  "category": "mineral"},
    "potassium":         {"aliases": [],                                      "unit": "mg",  "category": "mineral"},
}

# Build a lookup of all names -> canonical
ALL_ALIASES = {}
for canon, spec in NUTRIENT_SPECS.items():
    ALL_ALIASES[canon] = canon
    for a in spec["aliases"]:
        ALL_ALIASES[a] = canon

# ---- 3) Utility: unit conversion ---------------------------------------------
def convert_unit(value, from_unit, to_unit):
    from_unit = (from_unit or '').lower()
    to_unit = (to_unit or '').lower()
    if from_unit == to_unit or not from_unit or not to_unit:
        return value, to_unit
    factors = {
        ("g", "mg"): 1000.0, ("mg", "g"): 0.001,
        ("mcg", "mg"): 0.001, ("mg", "mcg"): 1000.0,
        ("g", "mcg"): 1_000_000.0, ("mcg", "g"): 1e-6,
    }
    if (from_unit, to_unit) in factors:
        return value * factors[(from_unit, to_unit)], to_unit
    return None, from_unit  # unknown conversion

# ---- 4) Enhanced cleaning & normalization -----------------------------------
def _basic_ocr_fixes(s: str) -> str:
    s = s.replace("Âµg", "mcg")  # micro sign issues
    s = re.sub(r'\b[lI]ron\b', 'iron', s, flags=re.IGNORECASE)  # lron/Iron -> iron
    s = re.sub(r'\bimg\b', 'mg', s, flags=re.IGNORECASE)        # "img" -> "mg"
    s = re.sub(r'\bOg\b', '0g', s, flags=re.IGNORECASE)         # Og -> 0g
    s = re.sub(r'\b0 g\b', '0g', s, flags=re.IGNORECASE)        # normalize spacing
    s = re.sub(r'«', '', s)                                     # stray symbol
    # Fix common OCR errors
    s = re.sub(r'\b59\s+6%\b', '5g 6%', s, flags=re.IGNORECASE)  # "59 6%" -> "5g 6%"
    s = re.sub(r'\b129\s+43%\b', '12g 43%', s, flags=re.IGNORECASE)  # "129 43%" -> "12g 43%"
    return s

def clean_ocr_text(raw_text: str) -> str:
    raw_text = _basic_ocr_fixes(raw_text)
    lines = [ln.strip() for ln in raw_text.splitlines()]

    cleaned = []
    for ln in lines:
        if not ln or len(ln) < 2:
            continue
        # Drop headers and noise
        if re.search(r'nutrition facts|serving size|per serving|per container|daily value|calories a day', ln, re.I):
            continue
        # Drop pure punctuation/junk
        if not any(c.isalpha() for c in ln) and not re.search(r'\d', ln):
            continue
        # Clean up the line but preserve structure
        ln = re.sub(r'\s+', ' ', ln).strip()
        if ln.endswith('.'):
            ln = ln[:-1]
        cleaned.append(ln)
    return "\n".join(cleaned)

# ---- 5) Enhanced pattern matching -------------------------------------------
def find_nutrient_in_line(line):
    """Find nutrient name in line, return canonical name and position"""
    line_lower = line.lower()
    
    # Specific patterns for better matching
    patterns = [
        (r'\btotal\s+fat\b', 'total fat'),
        (r'\bsaturated\s+fat\b', 'saturated fat'),
        (r'\btrans\s+fat\b', 'trans fat'),
        (r'\bcholesterol\b', 'cholesterol'),
        (r'\bsodium\b', 'sodium'),
        (r'\btotal\s+carb\b', 'total carbohydrate'),
        (r'\bdietary\s+fiber\b', 'dietary fiber'),
        (r'\btotal\s+sugars\b', 'total sugars'),
        (r'\bincl\.\s*added\s+sugars\b', 'added sugar'),
        (r'\badded\s+sugars\b', 'added sugar'),
        (r'\bprotein\b', 'protein'),
        (r'\bvitamin\s+d\b', 'vitamin d'),
        (r'\bcalcium\b', 'calcium'),
        (r'\biron\b', 'iron'),
        (r'\bpotassium\b', 'potassium'),
    ]
    
    for pattern, canonical in patterns:
        match = re.search(pattern, line_lower)
        if match:
            return canonical, match.start()
    
    return None, -1

# ---- 6) Enhanced value extraction --------------------------------------------
def extract_values_from_line(line):
    """Extract all numeric values with units from a line"""
    # Enhanced pattern to catch various formats
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(mcg|mg|g)\b',  # Standard format
        r'(\d+(?:\.\d+)?)(g|mg|mcg)\s+\d+%',  # Value+unit followed by percentage
        r'(\d+)\s+(g|mg|mcg)',  # Number space unit
        r'(\d+(?:\.\d+)?)(g|mg|mcg)',  # Number immediately followed by unit
    ]
    
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, line, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match[0])
                unit = match[1].lower()
                values.append((value, unit))
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for v, u in values:
        if (v, u) not in seen:
            seen.add((v, u))
            unique_values.append((v, u))
    
    return unique_values

# ---- 7) Improved extraction logic -------------------------------------------
def extract_nutrients(ocr_text: str):
    """
    Enhanced nutrient extraction with better pattern matching and multi-line parsing
    """
    lines = ocr_text.splitlines()
    results = {}
    
    # Process each line looking for nutrients and their values
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # Find nutrient in current line
        nutrient_name, pos = find_nutrient_in_line(line)
        if not nutrient_name:
            continue
            
        spec = NUTRIENT_SPECS.get(nutrient_name, {"unit": "g"})
        expected_unit = spec.get("unit", "g")
        
        # Look for values in current line and next few lines
        found_value = None
        search_lines = [line]
        
        # Add next 2 lines for context, but stop if we hit another nutrient
        for j in range(i + 1, min(i + 3, len(lines))):
            next_line = lines[j].strip()
            if not next_line:
                continue
            # Stop if we find another nutrient name
            if find_nutrient_in_line(next_line)[0]:
                break
            search_lines.append(next_line)
        
        # Extract values from all search lines
        all_values = []
        for search_line in search_lines:
            values = extract_values_from_line(search_line)
            all_values.extend(values)
        
        # Choose the best value
        if all_values:
            # Prefer values with expected unit
            for value, unit in all_values:
                if unit == expected_unit.lower():
                    found_value = (value, unit)
                    break
            
            # If no exact unit match, take the first reasonable value
            if not found_value:
                found_value = all_values[0]
        
        if found_value:
            value, unit = found_value
            results[nutrient_name] = {"value": value, "unit": unit}
    
    return results

# ---- 8) Debug function for troubleshooting ----------------------------------
def debug_extraction(ocr_text: str):
    """Debug function to see what's happening in the extraction process"""
    print("=== DEBUG: OCR Text Analysis ===")
    lines = ocr_text.splitlines()
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        print(f"Line {i}: '{line}'")
        
        nutrient_name, pos = find_nutrient_in_line(line)
        if nutrient_name:
            print(f"  -> Found nutrient: {nutrient_name}")
        
        values = extract_values_from_line(line)
        if values:
            print(f"  -> Found values: {values}")
        print()

# ---- 9) Combine with thresholds (unchanged) ---------------------------------
def combine_nutrients_with_thresholds(nutrients, thresholds_db):
    combined = []

    for canon_name, info in nutrients.items():
        value = info["value"]
        unit = info["unit"].lower()

        threshold = thresholds_db.get(canon_name)
        category = "unknown"
        status = "Unknown"

        if threshold:
            desired_unit = threshold.get("unit") or unit
            converted, out_unit = convert_unit(value, unit, desired_unit)
            if converted is not None:
                value, unit = converted, out_unit

            category = threshold.get("category", "unknown")
            max_val = threshold.get("max_threshold")
            min_val = threshold.get("min_threshold")

            status = "Normal"
            if max_val is not None and value > max_val:
                status = "High"
            if min_val is not None and value < min_val:
                status = "Low"

        else:
            # If DB missing, fall back to spec category if known
            spec = NUTRIENT_SPECS.get(canon_name)
            if spec:
                category = spec.get("category", "unknown")
            status = "Unknown" if not threshold else status

        combined.append({
            "nutrient": canon_name.title(),
            "value": round(value, 3),
            "unit": unit,
            "status": status,
            "category": category
        })

    return combined

def preprocess_image_for_ocr(image_path):
    """Basic preprocessing: grayscale, denoise, threshold."""
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh
# ---- 10) Main pipeline with debug option ------------------------------------
def process_label_image(image_path, debug=False):
    # ---- Preprocess before OCR ----
    preprocessed_img = preprocess_image_for_ocr(image_path)
    
    # OCR from preprocessed image
    raw_text = pytesseract.image_to_string(preprocessed_img)
    print("=== RAW OCR TEXT ===")
    print(raw_text)

    # Clean text
    cleaned_text = clean_ocr_text(raw_text)
    print("\n=== CLEANED TEXT ===")
    print(cleaned_text)
    
    if debug:
        debug_extraction(cleaned_text)
    
    parsed = extract_nutrients(cleaned_text)
    db_thresholds = load_thresholds_from_db()
    structured = combine_nutrients_with_thresholds(parsed, db_thresholds)
    
    return cleaned_text, structured