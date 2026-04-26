import sqlite3
from datetime import datetime, timedelta


# ------------------------------------------------------------------ #
#  Daily needs                                                         #
# ------------------------------------------------------------------ #

def calculate_daily_needs(weight_kg, activity_level):
    """
    Calculate daily macro targets from weight and activity level.
    Raises ValueError for invalid inputs.
    """
    if weight_kg is None:
        raise ValueError("Weight is required to calculate daily needs.")

    try:
        weight_kg = float(weight_kg)
    except (ValueError, TypeError):
        raise ValueError("Invalid weight value.")

    if weight_kg <= 0:
        raise ValueError("Weight must be greater than 0.")

    if not activity_level:
        raise ValueError("Activity level is required.")

    multiplier = {'sedentary': 25, 'moderate': 30, 'active': 35}.get(activity_level, 30)
    calories   = weight_kg * multiplier

    return {
        "calories":  round(calories),
        "protein_g": round(weight_kg * 1.2),
        "fats_g":    round((0.25 * calories) / 9),
        "carbs_g":   round((0.50 * calories) / 4),
    }


# ------------------------------------------------------------------ #
#  Nutrient status evaluation                                          #
# ------------------------------------------------------------------ #

_NAME_MAPPINGS = {
    'sugar':           ['sugar', 'added sugar', 'total sugar', 'total sugars'],
    'fat':             ['fat', 'total fat', 'fats'],
    'saturated fat':   ['saturated fat', 'sat fat'],
    'trans fat':       ['trans fat'],
    'sodium':          ['sodium', 'salt'],
    'carbohydrates':   ['carbohydrates', 'carbs', 'total carbohydrates', 'total carbohydrate'],
    'protein':         ['protein'],
    'fiber':           ['fiber', 'dietary fiber'],
    'cholesterol':     ['cholesterol'],
    'calcium':         ['calcium'],
    'iron':            ['iron'],
    'vitamin c':       ['vitamin c', 'ascorbic acid'],
    'vitamin a':       ['vitamin a'],
    'vitamin d':       ['vitamin d'],
    'potassium':       ['potassium'],
}


def evaluate_nutrient_status_enhanced(nutrient_name, value):
    """
    Look up threshold data in food_thresholds.db and return
    a dict with 'status' and 'message'.
    Falls back through alias mapping and LIKE search if exact match fails.
    """
    conn   = sqlite3.connect('food_thresholds.db')
    cursor = conn.cursor()

    def _query(name):
        cursor.execute("""
            SELECT max_threshold, min_threshold, high_risk_message, low_risk_message
            FROM ingredient_thresholds
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
        """, (name,))
        return cursor.fetchone()

    result = _query(nutrient_name)

    # Alias fallback
    if not result:
        for db_name, variations in _NAME_MAPPINGS.items():
            if nutrient_name.lower() in [v.lower() for v in variations]:
                result = _query(db_name)
                if result:
                    break

    # LIKE fallback
    if not result:
        cursor.execute("""
            SELECT max_threshold, min_threshold, high_risk_message, low_risk_message
            FROM ingredient_thresholds
            WHERE LOWER(name) LIKE LOWER(?)
        """, (f'%{nutrient_name}%',))
        result = cursor.fetchone()

    conn.close()

    if not result:
        return {
            "status":  "Unknown",
            "message": f"No threshold data available for {nutrient_name}.",
        }

    max_threshold, min_threshold, high_msg, low_msg = result

    if max_threshold is not None and value > max_threshold:
        return {"status": "High",   "message": high_msg}
    if min_threshold is not None and value < min_threshold:
        return {"status": "Low",    "message": low_msg}
    return {"status": "Normal", "message": "This level is within the healthy range."}


# ------------------------------------------------------------------ #
#  Health score                                                        #
# ------------------------------------------------------------------ #

_NUTRIENT_ALIASES = {
    'protein': ['protein'],
    'carbs':   ['carbohydrates', 'carbs', 'total carbohydrates'],
    'fats':    ['fat', 'fats', 'total fat'],
    'sodium':  ['sodium'],
    'sugar':   ['sugar', 'sugars', 'added sugar'],
}


def _get_nutrient_value(aggregated, key):
    for alias in _NUTRIENT_ALIASES.get(key, [key]):
        if alias in aggregated:
            return aggregated[alias]
    return 0


def calculate_health_score(aggregated_nutrients, daily_recommendations):
    """
    Score 0–100 based on how closely aggregated intake matches daily targets.
    Penalties for excess sodium and sugar are capped at 20 points each.
    """
    score = 100.0

    macro_goals = {
        'protein': daily_recommendations.get('protein_g', 1),
        'carbs':   daily_recommendations.get('carbs_g',   1),
        'fats':    daily_recommendations.get('fats_g',    1),
    }

    for macro, goal in macro_goals.items():
        actual    = _get_nutrient_value(aggregated_nutrients, macro)
        deviation = abs(actual - goal) / max(goal, 1)
        score    -= min(deviation * 40, 20)

    sodium = _get_nutrient_value(aggregated_nutrients, 'sodium')
    if sodium > 2300:
        score -= min(((sodium - 2300) / 2300) * 40, 20)

    sugar = _get_nutrient_value(aggregated_nutrients, 'sugar')
    if sugar > 50:
        score -= min(((sugar - 50) / 50) * 40, 20)

    return max(0, round(score))


# ------------------------------------------------------------------ #
#  Historical health scores (for charts)                              #
# ------------------------------------------------------------------ #

def get_historical_health_scores(username, period, recommendations, scan_collection):
    """
    Aggregate scan data over a rolling window and return per-period health scores.

    Parameters
    ----------
    username        : str
    period          : 'daily' | 'weekly' | 'monthly'
    recommendations : dict from calculate_daily_needs()
    scan_collection : pymongo Collection (scan_storage.collection)

    Returns
    -------
    {"labels": [...], "scores": [...]}
    """
    if scan_collection is None:
        return {"labels": [], "scores": []}

    end_date = datetime.utcnow()

    if period == 'daily':
        start_date   = end_date - timedelta(days=30)
        date_format  = "%Y-%m-%d"
        group_id     = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
        divisor      = 1
    elif period == 'weekly':
        start_date   = end_date - timedelta(weeks=12)
        date_format  = "%Y-%U"
        group_id     = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
        divisor      = 7
    else:  # monthly
        start_date   = end_date - timedelta(days=365)
        date_format  = "%Y-%m"
        group_id     = {"$dateToString": {"format": date_format, "date": "$scan_date"}}
        divisor      = 30

    pipeline = [
        {"$match": {"username": username, "scan_date": {"$gte": start_date}}},
        {"$unwind": "$nutrition_analysis.structured_nutrients"},
        {"$group": {
            "_id": {
                "period":   group_id,
                "nutrient": {"$toLower": "$nutrition_analysis.structured_nutrients.nutrient"},
            },
            "total_value": {"$sum": "$nutrition_analysis.structured_nutrients.value"},
        }},
        {"$group": {
            "_id":       "$_id.period",
            "nutrients": {"$push": {"k": "$_id.nutrient", "v": "$total_value"}},
        }},
        {"$addFields": {"nutrients": {"$arrayToObject": "$nutrients"}}},
        {"$sort": {"_id": 1}},
    ]

    results = list(scan_collection.aggregate(pipeline))

    labels, scores = [], []
    for result in results:
        aggregated = {k: v / divisor for k, v in result['nutrients'].items()}
        scores.append(calculate_health_score(aggregated, recommendations))
        labels.append(result['_id'])

    return {"labels": labels, "scores": scores}


# ------------------------------------------------------------------ #
#  AI result fuzzy key matching                                        #
# ------------------------------------------------------------------ #

def match_ai_result(ai_results, key):
    """
    Match a nutrient key against AI result keys.
    Priority: exact → substring → word overlap.
    Returns None if no match found.
    """
    if not ai_results:
        return None

    if key in ai_results:
        return ai_results[key]

    for ai_key, val in ai_results.items():
        if ai_key in key or key in ai_key:
            return val

    key_words = set(key.split())
    for ai_key, val in ai_results.items():
        if key_words & set(ai_key.split()):
            return val

    return None