from transformers import pipeline
import torch

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


# --- 3. INITIALIZE ZERO-SHOT CLASSIFICATION MODEL ---
try:
    print("🤖 Loading NLP Zero-Shot model (facebook/bart-large-mnli)...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ NLP model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load NLP model: {e}")
    print("⚠️ Falling back to keyword-based analysis.")
    classifier = None


# --- 4. TEXT ANALYSIS FUNCTION ---
def analyze_report_text(description: str) -> str:
    """
    Classifies a report description into one of the predefined categories
    using a zero-shot NLP model. Falls back to 'Other' if uncertain.
    """
    if not classifier:
        return "Other"

    try:
        result = classifier(description, TOPIC_LABELS, multi_label=False)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_score >= CONFIDENCE_THRESHOLD:
            print(f"🧠 NLP: '{description}' → '{top_label}' (Score: {top_score:.2f})")
            return top_label
        else:
            print(f"🤔 Low confidence ({top_score:.2f}) for '{description}'. Defaulting to 'Other'.")
            return "Other"

    except Exception as e:
        print(f"❌ NLP Error: {e}")
        return "Other"


# --- 5. EXPLANATORY INSIGHT GENERATOR ---
def generate_explanatory_insights(aggregated_data: dict) -> list:
    """
    Converts raw report counts into human-friendly insights.
    Each insight contains text, category, severity, and an icon.
    """
    insights = []
    for category, info in aggregated_data.items():
        count = info.get("count", 0)
        severity = info.get("severity", "low")

        if count == 0:
            continue

        # Frequency descriptor
        if count >= 10:
            prefix = "⚠️ Multiple users have reported"
        elif count >= 5:
            prefix = "⚠️ Several users mentioned"
        elif count >= 2:
            prefix = "ℹ️ A few users noticed"
        else:
            prefix = "ℹ️ One user reported"

        # Severity-based phrasing
        if severity == "high":
            suffix = "after consuming this product. Please exercise caution."
        elif severity == "medium":
            suffix = "after consuming this product. Be aware if you are sensitive."
        else:
            suffix = "after using this product, though it seems mild."

        # Readable category mapping
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


# --- 6. TESTING BLOCK ---
if __name__ == '__main__':
    print("\n--- 🧪 Testing NLP Analyzer ---\n")

    test_reports = [
        "threw up after eating this, my stomach hurts",
        "my skin got itchy and red",
        "tasted really weird and metallic",
        "found a piece of plastic inside",
        "this is actually delicious!",
        "the box was already open when i got it",
        "felt queasy and lightheaded"
    ]

    for report in test_reports:
        topic = analyze_report_text(report)
        print(f"📝 Report: {report}\n→ Topic: {topic}\n")
