import pytesseract
import re
import sqlite3
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_thresholds_from_db():
    conn = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, unit, max_threshold, min_threshold, category FROM ingredient_thresholds")
    rows = cursor.fetchall()
    conn.close()
    
    thresholds = {}
    for name, unit, max_val, min_val, category in rows:
        thresholds[name.lower()] = {
            "unit": unit,
            "max_threshold": max_val,
            "min_threshold": min_val,
            "category": category
        }
    return thresholds

def clean_ocr_text(raw_text):
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if line.isdigit() or not any(c.isalpha() for c in line):
            continue
        if any(char in line for char in ['@', '#', '*', '=', '~', '|']):
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def normalize_nutrient_name(name):
    name = name.lower().strip()
    aliases = {
        "total sugars": "sugar",
        "sugars": "sugar",
        "added sugar": "sugar",
        "salt": "sodium",
        "fat": "total fat",
        "total fats": "total fat",
        "trans fat": "trans fats",
        "cholesterol": "cholesterol",
        "carbs": "carbohydrates",
        "carbohydrate": "carbohydrates",
        "protein": "protein"
    }
    for alias, standard in aliases.items():
        if alias in name:
            return standard
    return name

def extract_nutrients(ocr_text):
    pattern = r'(?P<name>[a-zA-Z\s]+?)\s*(?P<value>\d+\.?\d*)\s*(?P<unit>mg|g)'
    matches = re.findall(pattern, ocr_text.lower())
    extracted = {}

    for name, value, unit in matches:
        norm_name = normalize_nutrient_name(name.strip())
        extracted[norm_name] = {"value": float(value), "unit": unit}
    return extracted

def combine_nutrients_with_thresholds(nutrients, thresholds_db):
    combined = []
    for nutrient, info in nutrients.items():
        status = "Normal"
        note = ""
        threshold = thresholds_db.get(nutrient)

        if threshold and info["unit"] == threshold["unit"]:
            max_val = threshold.get("max_threshold")
            min_val = threshold.get("min_threshold")

            if max_val is not None and info["value"] > max_val:
                status = "High"
            elif min_val is not None and info["value"] < min_val:
                status = "Low"
        
        combined.append({
            "nutrient": nutrient.title(),
            "value": info["value"],
            "unit": info["unit"],
            "status": status,
            "category": threshold["category"] if threshold else "unknown"
        })

    return combined

def process_label_image(image_path):
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    cleaned_text = clean_ocr_text(raw_text)
    nutrients = extract_nutrients(cleaned_text)
    db_thresholds = load_thresholds_from_db()
    structured_nutrients = combine_nutrients_with_thresholds(nutrients, db_thresholds)
    return cleaned_text, structured_nutrients
