"""
Gemini API integration for AI-powered formulation recommendations.

Best Use of Generative AI - Hack Duke 2026
"""

import json
from typing import Optional

GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass


def init_gemini(api_key: Optional[str] = None) -> bool:
    """Initialize Gemini with API key. Returns True if ready."""
    if not GEMINI_AVAILABLE:
        return False
    key = api_key or _get_key()
    if not key:
        return False
    try:
        genai.configure(api_key=key)
        return True
    except Exception:
        return False


def _get_key() -> Optional[str]:
    """Get API key from env or Streamlit secrets."""
    import os
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def ask_formulation_advice(
    protein_summary: str,
    composition: dict,
    stability_score: float,
    question: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Use Gemini to provide natural language formulation advice.
    """
    if not init_gemini(api_key):
        return "Gemini API key not configured. Add GEMINI_API_KEY to .env or Streamlit secrets."

    comp_str = ", ".join(f"{k}: {v:.2f}" for k, v in composition.items())
    prompt = f"""You are an expert in polymer-protein hybrid formulation for PET-RAFT chemistry.

Protein: {protein_summary}
Monomer composition: {comp_str}
Stability score: {stability_score:.3f} (0-1, higher = more favorable)

Available monomers: SPMA (anionic), TMAEMA/DEAEMA/DMAPMA (cationic), HPMA/PEGMA (hydrophilic), BMA/EHMA (hydrophobic).

User question: {question}

Provide a concise, actionable response (2-4 sentences) about formulation optimization."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text or "No response generated."
    except Exception as e:
        return f"API error: {str(e)}"
