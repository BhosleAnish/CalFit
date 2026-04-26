import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
#  Client setup                                                        #
# ------------------------------------------------------------------ #

try:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        client = openai.OpenAI(api_key=_api_key)
        print("✅ OpenAI client configured successfully.")
    else:
        client = None
        print("⚠️ OPENAI_API_KEY not found. AI features will be disabled.")
except Exception as e:
    print(f"⚠️ Could not configure OpenAI client: {e}")
    client = None


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _match_ai_result(ai_results, key):
    """
    Fuzzy-match a nutrient key against the keys returned by the AI.
    Priority: exact → substring → word overlap.
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


# ------------------------------------------------------------------ #
#  Batch nutrient analysis (single OpenAI call for all nutrients)     #
# ------------------------------------------------------------------ #

def get_personalized_nutrient_analysis_batch(nutrients, user_profile):
    """
    One OpenAI call for the full nutrient list.
    Returns a dict keyed by lowercased nutrient name.
    """
    if not client or not user_profile:
        return {
            n.get('nutrient', '').lower(): "Complete your profile for personalized insights."
            for n in nutrients
        }

    user_context = {
        "age":                user_profile.get('age', 'N/A'),
        "gender":             user_profile.get('gender', 'N/A'),
        "weight_kg":          user_profile.get('weight_kg', 'N/A'),
        "height_cm":          user_profile.get('height_cm', 'N/A'),
        "activity_level":     user_profile.get('activity_level', 'N/A'),
        "medical_conditions": user_profile.get('medical_conditions', 'None'),
        "allergies":          user_profile.get('allergies', 'None'),
    }

    nutrient_list = "\n".join([
        f"- {n.get('nutrient', '?')}: {n.get('value', 0)} {n.get('unit', 'g')} "
        f"(Status: {n.get('status', 'Unknown')})"
        for n in nutrients
    ])

    expected_keys = [n.get('nutrient', '').lower().strip() for n in nutrients]

    prompt = f"""You are an expert clinical nutritionist giving personalized, actionable dietary advice.

USER PROFILE:
- Age: {user_context['age']}
- Gender: {user_context['gender']}
- Weight: {user_context['weight_kg']} kg
- Height: {user_context['height_cm']} cm
- Activity Level: {user_context['activity_level']}
- Medical Conditions: {user_context['medical_conditions']}
- Allergies: {user_context['allergies']}

NUTRIENTS TO ANALYZE:
{nutrient_list}

INSTRUCTIONS — follow these rules strictly for each nutrient:
1. HIGH status: Explain the specific health risk this elevated level poses FOR THIS USER. Suggest one concrete dietary action to reduce it.
2. LOW status: Explain what this deficiency means for THIS USER's body. Recommend one specific food or habit to address it.
3. NORMAL status: Don't just say "it's fine". Explain WHY this level is beneficial for this specific person. Add one tip to maintain it.
4. UNKNOWN status: Do NOT say "unknown". Use the user's profile to estimate whether this level is adequate, borderline, or concerning FOR THEM, and explain with a brief recommendation.

TONE RULES:
- Be direct, specific, and actionable — never vague.
- Never just restate the status word (High/Low/Normal/Unknown).
- Always reference at least one detail from the user's profile.
- Each answer must be 1-2 sentences max.

Return ONLY valid JSON. No markdown, no explanation, no extra text:
{{
  "nutrient_name_lowercase": "personalized insight here",
  ...
}}

CRITICAL: Use EXACTLY these keys (lowercased, verbatim): {expected_keys}
Every key must appear in the response. Do NOT rename keys."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical nutritionist. Return only valid JSON with exactly the keys provided. "
                        "Never restate the status word. Always give specific, profile-aware, actionable insights. "
                        "No markdown fences. No explanation outside the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    except Exception as e:
        print(f"❌ Batch AI analysis failed: {e}")
        return {}


# ------------------------------------------------------------------ #
#  Single-nutrient wrapper (used by barcode / index route)            #
# ------------------------------------------------------------------ #

def get_personalized_nutrient_analysis(
    nutrient_name, nutrient_value, nutrient_unit, nutrient_status, user_profile
):
    """Convenience wrapper — calls the batch function with one nutrient."""
    if not client or not user_profile:
        return "AI analysis unavailable."

    result = get_personalized_nutrient_analysis_batch(
        [{"nutrient": nutrient_name, "value": nutrient_value,
          "unit": nutrient_unit, "status": nutrient_status}],
        user_profile,
    )
    return _match_ai_result(result, nutrient_name.lower()) or "Unable to generate analysis."


# ------------------------------------------------------------------ #
#  Comprehensive analysis (profile page AI report)                    #
# ------------------------------------------------------------------ #

def get_comprehensive_ai_analysis(username, scan_storage, profile_storage):
    """
    Generates a personalized, doctor-style HTML analysis
    across all of a user's scans.

    Parameters are passed in rather than read from session so this
    function stays stateless and testable.
    """
    if scan_storage.collection is None:
        return "<h2>AI Analysis Unavailable</h2><p>Database not connected.</p>"

    try:
        user_scans   = list(scan_storage.collection.find({"username": username}))
        user_profile = profile_storage.get_profile(username)
    except Exception as e:
        print(f"❌ Could not fetch data: {e}")
        return "<h2>Error</h2><p>Could not fetch user data.</p>"

    if not user_profile:
        return "<h2>Profile Required</h2><p>Please complete your profile for analysis.</p>"

    if not user_scans:
        return "<h2>No Data</h2><p>No scans found. Please scan items to get an analysis.</p>"

    # --- Aggregate nutrients across all scans ---
    total_nutrients = {}
    risk_counts     = {"High": 0, "Low": 0, "Normal": 0, "Unknown": 0}

    for scan in user_scans:
        for nutrient in scan.get("nutrition_analysis", {}).get("structured_nutrients", []):
            name   = nutrient.get("nutrient", "").lower()
            value  = nutrient.get("value", 0)
            status = nutrient.get("status", "Unknown")

            if name:
                if name not in total_nutrients:
                    total_nutrients[name] = {"total": 0, "count": 0, "unit": nutrient.get("unit", "")}
                total_nutrients[name]["total"] += value
                total_nutrients[name]["count"] += 1

            if status in risk_counts:
                risk_counts[status] += 1

    avg_nutrients = {
        name: {
            "average": round(data["total"] / data["count"], 2),
            "unit":    data["unit"],
            "total":   round(data["total"], 2),
        }
        for name, data in total_nutrients.items()
        if data["count"] > 0
    }

    analysis_data = {
        "user_profile": {
            "name":           user_profile.get("full_name", "User"),
            "age":            user_profile.get("age"),
            "gender":         user_profile.get("gender"),
            "activity_level": user_profile.get("activity_level"),
        },
        "risk_distribution":  risk_counts,
        "average_nutrients":  avg_nutrients,
    }

    # Only add daily recommendations if profile is complete enough
    if user_profile.get("weight_kg") and user_profile.get("activity_level"):
        try:
            from utils.nutrition import calculate_daily_needs
            analysis_data["daily_recommendations"] = calculate_daily_needs(
                user_profile["weight_kg"], user_profile["activity_level"]
            )
        except Exception as e:
            print(f"⚠️ Could not calculate daily needs: {e}")

    data_json = json.dumps(analysis_data, indent=2)

    prompt = f"""
You are an expert nutritionist providing a personalized health analysis. Analyze the following data:

```json
{data_json}
```

Generate your full response in **HTML format** with exactly these 5 sections:
1. <h3>User Profile Overview</h3> — summarize user's basic info and context.
2. <h3>Health Risk Distribution</h3> — explain what the High, Normal, Low, and Unknown counts mean.
   For each category, give one explanatory sentence.
3. <h3>Nutrient & Daily Recommendation Insights</h3> — merge both analyses here.
   Present key nutrients as bullet points, each being a 1-sentence observation comparing
   user's average intake vs daily needs, written like a doctor's remark.
   For every nutrient: include the average intake, the recommended intake, and a brief interpretation.
   Example:
   <li>Average <strong>Calories</strong> intake is 478 kcal, compared to the recommended 1300 kcal — indicating lower energy intake than ideal.</li>
4. <h3>Overall Health Summary</h3> — 2–3 sentences on the user's overall diet quality.
5. <h3>Next Steps & Suggestions</h3> — 3–4 bullet points with actionable guidance.

Formatting rules:
- Each section MUST be wrapped in <div class="analysis-section">.
- Use <p> for normal text, <ul>/<li> for bullet points, <strong> for key terms.
- DO NOT include <html>, <body>, or markdown fences.
"""

    if not client:
        return "<h2>AI Analysis Unavailable</h2><p>OpenAI client not configured.</p>"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a qualified nutritionist who writes professional, structured HTML reports. "
                        "Provide context, avoid numbers alone, and make insights sound like doctor explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=2000,
        )

        raw_html = response.choices[0].message.content.strip()
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
        return raw_html.strip()

    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return "<h2>Analysis Error</h2><p>Could not generate AI analysis. Please try again.</p>"