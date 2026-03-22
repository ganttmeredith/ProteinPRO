"""
Auth0 configuration for optional user authentication.

Enables saving formulations to user profiles and PDB cache.
Uses Streamlit native st.login() with OIDC when configured.
"""

from typing import Optional


def _get_config() -> dict:
    import os
    cfg = {}
    cfg["domain"] = os.environ.get("AUTH0_DOMAIN", "").strip()
    cfg["client_id"] = os.environ.get("AUTH0_CLIENT_ID", "").strip()
    cfg["client_secret"] = os.environ.get("AUTH0_CLIENT_SECRET", "").strip()
    cfg["audience"] = os.environ.get("AUTH0_AUDIENCE", "")
    if not cfg["domain"]:
        try:
            import streamlit as st
            s = st.secrets
            if hasattr(s, "auth") and hasattr(s.auth, "auth0"):
                a = s.auth.auth0
                meta = getattr(a, "server_metadata_url", "")
                if meta:
                    cfg["domain"] = str(meta).replace("https://", "").split("/")[0]
                cfg["client_id"] = getattr(a, "client_id", "") or ""
                cfg["client_secret"] = getattr(a, "client_secret", "") or ""
        except Exception:
            pass
    return cfg


def is_available() -> bool:
    """Check if Auth0 is configured (env or secrets)."""
    cfg = _get_config()
    return bool(cfg.get("domain") and cfg.get("client_id"))


def is_logged_in() -> bool:
    """Check if user is logged in via Streamlit auth."""
    try:
        import streamlit as st
        return getattr(st.user, "is_logged_in", False)
    except Exception:
        return False


def get_user_id() -> Optional[str]:
    """Get current user ID (sub claim) if logged in."""
    try:
        import streamlit as st
        if getattr(st.user, "is_logged_in", False):
            return getattr(st.user, "sub", None) or getattr(st.user, "email", None) or "unknown"
    except Exception:
        pass
    return None
