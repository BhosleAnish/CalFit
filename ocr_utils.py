import pytesseract
import json
import re
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_risk_database(path="food_rules.json"):
    with open(path, 'r') as file:
        return json.load(file)

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

def assess_risks(nutrients, db):
    flagged = []
    for nutrient, info in nutrients.items():
        if nutrient in db:
            rule = db[nutrient]
            if info["unit"] == rule["unit"] and info["value"] > rule["threshold"]:
                flagged.append({
                    "nutrient": nutrient,
                    "value": info["value"],
                    "unit": info["unit"],
                    "advice": rule["advice"]
                })
    return flagged

def process_label_image(image_path, rules_path="food_rules.json"):
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    cleaned_text = clean_ocr_text(raw_text)
    nutrients = extract_nutrients(cleaned_text)
    rule_db = load_risk_database(rules_path)
    risks = assess_risks(nutrients, rule_db)
    return cleaned_text, risks
