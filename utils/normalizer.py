import uuid
from datetime import datetime

def normalize_data(raw_data, source="ocr", username=None, image_filename=None):
    """
    Convert raw OCR/API output into a universal schema for MongoDB storage.
    """

    return {
        "product_id": str(uuid.uuid4()),
        "username": username,
        "product_name": raw_data.get("product_name", "Unknown Product"),
        "scan_date": datetime.utcnow(),
        "source": source,

        "ingredients": [
            {
                "name": ing.get("name") if isinstance(ing, dict) else ing,
                "confidence": ing.get("confidence") if isinstance(ing, dict) else None
            }
            for ing in raw_data.get("ingredients", [])
        ],

        "nutrients": [
            {
                "name": n.get("name"),
                "value": n.get("value"),
                "unit": n.get("unit")
            }
            for n in raw_data.get("nutrients", [])
        ],

        "analysis": raw_data.get("analysis", []),

        "scan_metadata": {
            "image_filename": image_filename,
            "processing_timestamp": datetime.utcnow().isoformat()
        },

        "system_info": {
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30),
            "app_version": "1.0"
        }
    }
