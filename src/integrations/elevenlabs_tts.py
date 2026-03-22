"""
ElevenLabs text-to-speech for accessibility.

Best Accessibility Hack - Hack Duke 2026
"""

from typing import Optional


def _get_key() -> Optional[str]:
    import os
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ELEVENLABS_API_KEY")
    except Exception:
        return None


def text_to_speech_audio(text: str, api_key: Optional[str] = None) -> Optional[bytes]:
    """
    Convert text to speech via ElevenLabs API. Returns audio bytes or None.
    """
    key = api_key or _get_key()
    if not key:
        return None

    try:
        import requests
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
        r = requests.post(url, json={"text": text[:500]}, headers=headers, timeout=15)
        return r.content if r.ok else None
    except Exception:
        return None


def is_available() -> bool:
    return bool(_get_key())
