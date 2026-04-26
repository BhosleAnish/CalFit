import re
from functools import wraps
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash


# ------------------------------------------------------------------ #
#  Route decorators                                                    #
# ------------------------------------------------------------------ #

def login_required(f):
    """Redirect unauthenticated users to the landing page."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated


def prevent_cache(f):
    """Add no-cache headers to the response to block browser back-button access."""
    @wraps(f)
    def decorated(*args, **kwargs):
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers['Cache-Control'] = (
                'no-cache, no-store, must-revalidate, private, max-age=0'
            )
            response.headers['Pragma']  = 'no-cache'
            response.headers['Expires'] = '0'
        return response
    return decorated


# ------------------------------------------------------------------ #
#  Password policy                                                     #
# ------------------------------------------------------------------ #

_POLICY = {
    "min_length":      8,
    "require_upper":   True,
    "require_lower":   True,
    "require_digit":   True,
    "require_special": True,
}

_SPECIAL_CHARS = r"[!@#$%^&*()\-_=+\[\]{};':\"\\|,.<>/?`~]"


def validate_password(password: str) -> tuple[bool, list[str]]:
    """
    Validate a password against the policy.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    if len(password) < _POLICY["min_length"]:
        errors.append(f"Password must be at least {_POLICY['min_length']} characters long.")

    if _POLICY["require_upper"] and not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter (A–Z).")

    if _POLICY["require_lower"] and not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter (a–z).")

    if _POLICY["require_digit"] and not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one number (0–9).")

    if _POLICY["require_special"] and not re.search(_SPECIAL_CHARS, password):
        errors.append("Password must contain at least one special character (!@#$%^&* etc.).")

    return (len(errors) == 0, errors)