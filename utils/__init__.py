from .auth import login_required, prevent_cache, validate_password
from .nutrition import (
    calculate_daily_needs,
    evaluate_nutrient_status_enhanced,
    calculate_health_score,
    get_historical_health_scores,
    match_ai_result,
)