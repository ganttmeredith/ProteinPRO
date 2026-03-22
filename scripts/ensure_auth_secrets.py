"""
Bootstrap: Create .streamlit/secrets.toml for Auth0 from .env if not present.

Run before first `streamlit run app.py` when using Auth0.
Alternatively, app.py calls this automatically at startup.
"""

import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

def ensure_auth_secrets() -> bool:
    """Create secrets.toml with Auth0 config from .env. Returns True if created/updated."""
    domain = os.environ.get("AUTH0_DOMAIN", "").strip()
    client_id = os.environ.get("AUTH0_CLIENT_ID", "").strip()
    client_secret = os.environ.get("AUTH0_CLIENT_SECRET", "").strip()
    if not domain or not client_id or not client_secret:
        return False

    streamlit_dir = Path(__file__).parent.parent / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    secrets_path = streamlit_dir / "secrets.toml"

    redirect_uri = os.environ.get("AUTH0_REDIRECT_URI", "http://localhost:8501/oauth2callback")
    cookie_secret = os.environ.get("AUTH0_COOKIE_SECRET", "change-me-use-random-string-32-chars")

    server_metadata_url = f"https://{domain}/.well-known/openid-configuration"

    auth_block = f'''# Auth0 - auto-generated from .env
[auth]
redirect_uri = "{redirect_uri}"
cookie_secret = "{cookie_secret}"

[auth.auth0]
client_id = "{client_id}"
client_secret = "{client_secret}"
server_metadata_url = "{server_metadata_url}"
'''

    # Only write if different or doesn't exist
    if not secrets_path.exists() or secrets_path.read_text() != auth_block:
        secrets_path.write_text(auth_block)
        return True
    return False

if __name__ == "__main__":
    if ensure_auth_secrets():
        print("Created .streamlit/secrets.toml for Auth0. Restart Streamlit if running.")
    else:
        print("Auth0 secrets not created (missing AUTH0_DOMAIN, AUTH0_CLIENT_ID, or AUTH0_CLIENT_SECRET in .env)")
