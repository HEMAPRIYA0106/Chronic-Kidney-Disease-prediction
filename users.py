import hashlib
import os

# ─────────────────────────────────────────────────────────────
#  Single-user store
#  Default credentials:  username = admin   password = admin123
#  Change them by calling set_credentials() once at startup,
#  or just edit HASHED_PASSWORD below after running:
#      python3 -c "import users; print(users._hash('yourpassword'))"
# ─────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    """SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()


# ── Default credentials (change these!) ──────────────────────
_USERNAME        = "admin"
_HASHED_PASSWORD = _hash("admin123")   # ← change 'admin123' to your password


def check_credentials(username: str, password: str) -> bool:
    """Return True if username + password match."""
    return (
        username.strip() == _USERNAME and
        _hash(password)  == _HASHED_PASSWORD
    )


def set_credentials(username: str, password: str):
    """Call this to change credentials at runtime."""
    global _USERNAME, _HASHED_PASSWORD
    _USERNAME        = username.strip()
    _HASHED_PASSWORD = _hash(password)