import os
import time
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

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
CONFIDENCE_THRESHOLD = 0.4

# --- 3. IN-MEMORY CLASSIFICATION CACHE ---
_classification_cache = {}

# --- 4. HuggingFace Inference API CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

if not HF_TOKEN:
    print("⚠️  HF_TOKEN not set. NLP classification will fail. Add it to your environment variables.")
else:
    print("✅ HuggingFace token loaded. Using Inference API for zero-shot classification.")


# --- 5. CALL HuggingFace INFERENCE API ---
def _call_hf_api(description: str, retries: int = 5) -> dict | None:
    """
    Calls the HuggingFace Inference API with retry logic for cold starts.
    - timeout=90s  : free tier model can take 30-45s to wake up
    - retries=5    : more attempts to survive intermittent failures
    - sleep=5s     : give the model breathing room between retries
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
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)

            if response.status_code == 503:
                wait_time = response.json().get("estimated_time", 20)
                print(f"⏳ HF model loading, waiting {wait_time:.0f}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"⚠️  HF API timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"❌ HF API request error: {e}")
            return None

    print("❌ HF API failed after all retries.")
    return None


# --- 6. PARSE API RESPONSE ---
def _parse_response(result) -> tuple:
    """Handles both singular and plural key formats from HF API."""
    if isinstance(result, list):
        result = result[0]
    top_label = result.get("label") or result["labels"][0]
    top_score = result.get("score") or result["scores"][0]
    return top_label, top_score


# --- 7. CLASSIFY A SINGLE UNIQUE DESCRIPTION (cache-aware) ---
def _classify_one_unique(desc: str) -> str:
    """
    Classifies a single description, checking the in-memory cache first.
    All API calls go through here so caching is always applied.
    """
    if desc in _classification_cache:
        print(f"💾 Cache hit: '{desc[:50]}'")
        return _classification_cache[desc]

    result = _call_hf_api(desc)
    if not result:
        return "Other"

    try:
        top_label, top_score = _parse_response(result)
        label = top_label if top_score >= CONFIDENCE_THRESHOLD else "Other"
        _classification_cache[desc] = label
        print(f"🧠 NLP: '{desc[:50]}' → '{label}' (Score: {top_score:.2f})")
        return label
    except (KeyError, IndexError) as e:
        print(f"❌ Parse error: {e}")
        return "Other"


# --- 8. SINGLE TEXT ANALYSIS (backward compatible with app.py) ---
def analyze_report_text(description: str) -> str:
    """
    Classifies a single report description into one of the predefined categories.
    Uses cache — repeated calls with the same text cost zero API calls.
    """
    if not description or not description.strip():
        return "Other"
    return _classify_one_unique(description.strip())


# --- 9. BATCH ANALYSIS (sequential on free tier to avoid competing for one model instance) ---
def analyze_reports_batch(descriptions: list) -> list:
    """
    Classifies reports with two key optimizations:

    1. Cache     : descriptions seen before skip the API entirely (instant).
    2. Dedup     : identical descriptions make only ONE API call regardless
                   of how many times they appear in the list.
    3. Sequential: max_workers=1 because the HF free tier runs a single model
                   instance. Parallel calls compete for it — one succeeds while
                   the other times out. Sequential calls each succeed reliably,
                   and with caching the second load is always instant anyway.

    Cold first load  : ~35s total (model wakeup + sequential calls)
    Cached load      : <0.01s (no API calls at all)
    """
    if not descriptions:
        return []

    stripped = [d.strip() for d in descriptions]
    unique_uncached = list({d for d in stripped if d and d not in _classification_cache})

    print(f"📊 Batch: {len(stripped)} total | {len(unique_uncached)} unique uncached → firing {len(unique_uncached)} API calls (sequential)")

    # max_workers=1 — sequential on purpose for free-tier HF single instance
    if unique_uncached:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(_classify_one_unique, desc): desc for desc in unique_uncached}
            for future in as_completed(futures):
                future.result()

    return [_classification_cache.get(d, "Other") if d else "Other" for d in stripped]


# --- 10. EXPLANATORY INSIGHT GENERATOR ---
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


# --- 11. TESTING BLOCK ---
if __name__ == '__main__':
    import time as t

    print("\n--- 🧪 Testing NLP Analyzer ---\n")

    test_reports = [
        "threw up after eating this, my stomach hurts",
        "my skin got itchy and red",
        "tasted really weird and metallic",
        "found a piece of plastic inside",
        "this is actually delicious!",
        "the box was already open when i got it",
        "felt queasy and lightheaded",
        # Duplicates — should cost zero extra API calls
        "threw up after eating this, my stomach hurts",
        "my skin got itchy and red",
    ]

    print("--- First batch (cold, 7 unique uncached) ---")
    start = t.time()
    topics = analyze_reports_batch(test_reports)
    print(f"\n✅ Done in {t.time() - start:.1f}s\n")
    for report, topic in zip(test_reports, topics):
        print(f"📝 {report}\n→ {topic}\n")

    print("\n--- Second batch (all cached, should be instant) ---")
    start = t.time()
    topics2 = analyze_reports_batch(test_reports)
    print(f"✅ Done in {t.time() - start:.3f}s (should be <0.01s)\n")