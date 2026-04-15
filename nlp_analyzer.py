import os
import time
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()  # Load .env file before reading HF_TOKEN

# --- 1. DEFINE TOPIC LABELS ---
TOPIC_LABELS = [
    "Sickness / Nausea / Vomiting",
    "Allergic Reaction / Rash / Itching",
    "Bad Taste / Foul Smell",
    "Packaging Defect / Foreign Object",
    "Headache / Dizziness",
    "Positive Feedback / No Issue"
]

# --- 2. SET CONFIDENCE THRESHOLD ---
CONFIDENCE_THRESHOLD = 0.4  # Ignore weak predictions

# --- 3. HuggingFace Inference API CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Render's Environment Variables
MODEL = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

if not HF_TOKEN:
    print("⚠️  HF_TOKEN not set. NLP classification will fail. Add it to your environment variables.")
else:
    print("✅ HuggingFace token loaded. Using Inference API for zero-shot classification.")


# --- 4. CALL HuggingFace INFERENCE API (with cold-start retry) ---
def _call_hf_api(description: str, retries: int = 3) -> dict | None:
    """
    Calls the HuggingFace Inference API with retry logic for cold starts.
    The free tier may take 10-30s to wake up on the first request.
    """
    if not HF_TOKEN:
        return None

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": description,
        "parameters": {
            "candidate_labels": TOPIC_LABELS,
            "multi_label": False
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=25)

            # Model is still loading (cold start) — wait and retry
            if response.status_code == 503:
                wait_time = response.json().get("estimated_time", 20)
                print(f"⏳ HF model loading, waiting {wait_time:.0f}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"⚠️  HF API timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"❌ HF API request error: {e}")
            return None

    print("❌ HF API failed after all retries.")
    return None


# --- 5. PARSE API RESPONSE ---
def _parse_response(result) -> tuple:
    """Handles both singular and plural key formats from HF API."""
    if isinstance(result, list):
        result = result[0]
    top_label = result.get("label") or result["labels"][0]
    top_score = result.get("score") or result["scores"][0]
    return top_label, top_score


# --- 6. SINGLE TEXT ANALYSIS (kept for backward compatibility with app.py) ---
def analyze_report_text(description: str) -> str:
    """
    Classifies a single report description into one of the predefined categories.
    Drop-in replacement — app.py requires no changes.
    For multiple reports, use analyze_reports_batch() for much better performance.
    """
    if not description or not description.strip():
        return "Other"

    result = _call_hf_api(description.strip())

    if not result:
        print("⚠️  NLP API unavailable. Returning 'Other'.")
        return "Other"

    try:
        top_label, top_score = _parse_response(result)

        if top_score >= CONFIDENCE_THRESHOLD:
            print(f"🧠 NLP: '{description}' → '{top_label}' (Score: {top_score:.2f})")
            return top_label
        else:
            print(f"🤔 Low confidence ({top_score:.2f}) for '{description}'. Defaulting to 'Other'.")
            return "Other"

    except (KeyError, IndexError) as e:
        print(f"❌ Unexpected API response format: {e} | Response: {result}")
        return "Other"


# --- 7. BATCH PARALLEL ANALYSIS (use this in get_community_warnings) ---
def analyze_reports_batch(descriptions: list) -> list:
    """
    Classifies ALL reports in PARALLEL — all API calls fire simultaneously.
    10 reports that used to take 50s now take ~5s total.

    Usage in app.py inside get_community_warnings():

        valid_reports = [(r.get("username"), r.get("description","").strip())
                         for r in all_reports
                         if r.get("username") and r.get("description","").strip()]

        descriptions = [desc for _, desc in valid_reports]
        topics = analyze_reports_batch(descriptions)

        for (username, desc), topic in zip(valid_reports, topics):
            if topic not in topic_users:
                topic_users[topic] = set()
            topic_users[topic].add(username)
    """
    if not descriptions:
        return []

    results = ["Other"] * len(descriptions)

    def classify_one(index, desc):
        result = _call_hf_api(desc)
        if not result:
            return index, "Other"
        try:
            top_label, top_score = _parse_response(result)
            label = top_label if top_score >= CONFIDENCE_THRESHOLD else "Other"
            print(f"🧠 NLP [{index}]: '{desc[:40]}' → '{label}' (Score: {top_score:.2f})")
            return index, label
        except (KeyError, IndexError) as e:
            print(f"❌ Parse error for report {index}: {e}")
            return index, "Other"

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(classify_one, i, desc): i
                   for i, desc in enumerate(descriptions)}
        for future in as_completed(futures):
            index, label = future.result()
            results[index] = label

    return results


# --- 8. EXPLANATORY INSIGHT GENERATOR (unchanged) ---
def generate_explanatory_insights(aggregated_data: dict) -> list:
    """
    Converts raw report counts into human-friendly insights.
    """
    insights = []
    for category, info in aggregated_data.items():
        count = info.get("count", 0)
        severity = info.get("severity", "low")

        if count == 0:
            continue

        if count >= 10:
            prefix = "⚠️ Multiple users have reported"
        elif count >= 5:
            prefix = "⚠️ Several users mentioned"
        elif count >= 2:
            prefix = "ℹ️ A few users noticed"
        else:
            prefix = "ℹ️ One user reported"

        if severity == "high":
            suffix = "after consuming this product. Please exercise caution."
        elif severity == "medium":
            suffix = "after consuming this product. Be aware if you are sensitive."
        else:
            suffix = "after using this product, though it seems mild."

        category_map = {
            "Sickness / Nausea / Vomiting": "nausea or stomach discomfort",
            "Allergic Reaction / Rash / Itching": "allergic reactions such as rash or itching",
            "Bad Taste / Foul Smell": "an unpleasant taste or smell",
            "Packaging Defect / Foreign Object": "packaging defects or foreign materials",
            "Headache / Dizziness": "headache or dizziness",
            "Positive Feedback / No Issue": "positive experiences and no issues reported",
        }

        readable_category = category_map.get(category, category.lower())
        insights.append({
            "text": f"{prefix} {readable_category} {suffix}",
            "category": category,
            "severity": severity,
            "icon": "⚠️" if severity != "low" else "ℹ️"
        })

    return insights


# --- 9. TESTING BLOCK ---
if __name__ == '__main__':
    import time as t

    print("\n--- 🧪 Testing NLP Analyzer (Parallel Batch) ---\n")

    test_reports = [
        "threw up after eating this, my stomach hurts",
        "my skin got itchy and red",
        "tasted really weird and metallic",
        "found a piece of plastic inside",
        "this is actually delicious!",
        "the box was already open when i got it",
        "felt queasy and lightheaded"
    ]

    start = t.time()
    topics = analyze_reports_batch(test_reports)
    elapsed = t.time() - start

    print(f"\n✅ All {len(test_reports)} reports classified in {elapsed:.1f}s\n")
    for report, topic in zip(test_reports, topics):
        print(f"📝 {report}\n→ Topic: {topic}\n")