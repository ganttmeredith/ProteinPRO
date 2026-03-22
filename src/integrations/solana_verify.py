"""
Solana integration for formulation verification / IP anchoring.

Best Use of Solana - Hack Duke 2026
Creates a hash of formulation + protein for on-chain verification (demo).
"""

import hashlib
import json
from typing import Optional


def formulation_hash(pdb_id: str, composition: dict, stability_score: float) -> str:
    """Create deterministic hash of formulation for verification."""
    payload = json.dumps({"pdb": pdb_id, "composition": composition, "score": stability_score}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_solana_address(api_key: Optional[str] = None) -> Optional[str]:
    """Get Solana wallet/address for demo. In production, connect wallet."""
    key = api_key or _get_key()
    if key:
        return key[:8] + "..." + key[-4:] if len(key) > 12 else key
    return None


def _get_key() -> Optional[str]:
    import os
    key = os.environ.get("SOLANA_PRIVATE_KEY") or os.environ.get("SOLANA_WALLET")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("SOLANA_PRIVATE_KEY")
    except Exception:
        return None


def is_available() -> bool:
    """Solana SDK optional; we use hash-based verification that works without it."""
    return True  # Hash verification always works
