import time
from .nlp_analyzer import analyze_reports_batch

# ------------------------------------------------------------------ #
#  In-memory cache                                                     #
# ------------------------------------------------------------------ #

_cache: dict = {}
CACHE_TTL = 300  # seconds


def get_community_warnings_cached(product_name, scan_storage):
    """Return cached community warnings, refreshing after TTL expires."""
    now = time.time()
    if product_name in _cache:
        ts, result = _cache[product_name]
        if now - ts < CACHE_TTL:
            print(f"💾 Community cache hit for '{product_name}'")
            return result

    result = get_community_warnings(product_name, scan_storage)
    _cache[product_name] = (now, result)
    return result


# ------------------------------------------------------------------ #
#  Core logic                                                          #
# ------------------------------------------------------------------ #

def get_community_warnings(product_name, scan_storage):
    """
    Analyse all user feedback for a product.
    Returns aggregated warnings, insights, and overall sentiment.
    Counts unique users per topic to prevent a single user skewing results.
    """
    print(f"DEBUG: Looking for reports for product: '{product_name}'")

    _empty = lambda msg: {
        "total_reports":    0,
        "insights":         [],
        "product_name":     product_name,
        "warning_level":    "none",
        "positive_reports": 0,
        "summary_message":  msg,
    }

    if not product_name or product_name == "Scanned from Label (OCR)":
        return _empty("No community data available yet.")

    if scan_storage.collection is None:
        print("ERROR: No database connection.")
        return _empty("Unable to retrieve community data.")

    try:
        pipeline = [
            {"$match": {
                "product_info.name": product_name,
                "user_feedback.issue_report.description": {"$exists": True, "$ne": ""},
            }},
            {"$project": {
                "username":    1,
                "description": "$user_feedback.issue_report.description",
            }},
        ]

        all_reports = list(scan_storage.collection.aggregate(pipeline))
        total_reports_count = len(all_reports)
        print(f"DEBUG: Found {total_reports_count} reports for '{product_name}'")

        if total_reports_count == 0:
            return _empty("No community reports found for this product.")

        # --- Classify reports with NLP ---
        valid_reports = [
            (r.get("username"), r.get("description", "").strip())
            for r in all_reports
            if r.get("username") and r.get("description", "").strip()
        ]

        descriptions = [desc for _, desc in valid_reports]
        topics       = analyze_reports_batch(descriptions)

        topic_users: dict[str, set] = {}
        for (username, _), topic in zip(valid_reports, topics):
            topic_users.setdefault(topic, set()).add(username)

        topic_counts = {topic: len(users) for topic, users in topic_users.items()}
        print(f"DEBUG: Unique user topic counts: {topic_counts}")

        # --- Known negative issue categories ---
        NEGATIVE_TOPICS = {
            "Sickness / Nausea / Vomiting":        "nausea, vomiting, or stomach discomfort",
            "Allergic Reaction / Rash / Itching":  "allergic reactions like rashes or itching",
            "Bad Taste / Foul Smell":              "bad taste or foul smell",
            "Packaging Defect / Foreign Object":   "packaging issues or foreign objects",
            "Headache / Dizziness":                "headaches, dizziness, or fatigue",
        }

        positive_count    = topic_counts.get("Positive Feedback / No Issue", 0)
        negative_reports  = {k: v for k, v in topic_counts.items() if k in NEGATIVE_TOPICS}
        total_neg_users   = sum(negative_reports.values())

        all_reporting_users = {u for users in topic_users.values() for u in users}
        total_unique        = len(all_reporting_users)

        neg_pct = (total_neg_users / total_unique * 100) if total_unique else 0

        # --- Warning level ---
        if   total_neg_users >= 5 and neg_pct >= 70: warning_level = "high"
        elif total_neg_users >= 3 and neg_pct >= 40: warning_level = "medium"
        elif total_neg_users >= 1:                   warning_level = "low"
        else:                                        warning_level = "none"

        print(f"WARNING LEVEL: {warning_level} ({total_neg_users}/{total_unique}, {neg_pct:.1f}%)")

        # --- Build human-readable insights ---
        insights = []
        for topic, count in sorted(negative_reports.items(), key=lambda x: x[1], reverse=True):
            desc = NEGATIVE_TOPICS[topic]
            if   count >= 10: prefix, suffix, severity = "Many users have reported",   "This issue seems widespread.",      "widespread"
            elif count >= 5:  prefix, suffix, severity = "Several users mentioned",    "It could be a recurring issue.",    "common"
            elif count >= 2:  prefix, suffix, severity = "A few users noticed",        "May not affect everyone.",          "some"
            else:             prefix, suffix, severity = "One user reported",          "This may be an isolated incident.", "few"

            insights.append({
                "text":     f"{prefix} {desc}. {suffix}",
                "severity": severity,
                "category": topic,
            })

        # --- Summary message ---
        if warning_level == "high":
            summary = f"HIGH RISK: {total_neg_users} of {total_unique} unique users reported major issues."
        elif warning_level == "medium":
            summary = "Moderate Risk: Some users reported recurring issues. Review before consumption."
        elif warning_level == "low":
            summary = "Low Risk: A few isolated reports exist, but overall community sentiment is neutral."
        elif positive_count >= total_unique * 0.7:
            summary = "Mostly Positive: Majority of users reported a good experience."
        else:
            summary = "No significant feedback trends detected."

        return {
            "total_reports":          total_reports_count,
            "total_unique_reporters": total_unique,
            "negative_reports":       total_neg_users,
            "positive_reports":       positive_count,
            "insights":               insights,
            "product_name":           product_name,
            "warning_level":          warning_level,
            "summary_message":        summary,
        }

    except Exception as e:
        import traceback
        print(f"ERROR in get_community_warnings: {e}")
        traceback.print_exc()
        return {
            "total_reports":    0,
            "insights":         [],
            "product_name":     product_name,
            "warning_level":    "none",
            "positive_reports": 0,
            "summary_message":  "Error retrieving community warnings.",
        }