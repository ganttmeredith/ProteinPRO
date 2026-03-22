"""
User-specific PDB file cache for logged-in users.

Saves uploaded and fetched PDB structures per user for persistence across sessions.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

CACHE_ROOT = Path(".pdb_cache")
USER_CACHE_SUBDIR = "users"


def _safe_user_id(user_id: str) -> str:
    """Sanitize user ID for use as directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))[:64]


def get_user_cache_dir(user_id: str) -> Path:
    """Get or create the cache directory for a user."""
    safe_id = _safe_user_id(user_id)
    path = CACHE_ROOT / USER_CACHE_SUBDIR / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fetched_to_user_cache(user_id: str, pdb_id: str, source_path: str) -> str:
    """Copy a fetched PDB file into the user's cache. Returns new path."""
    cache_dir = get_user_cache_dir(user_id)
    pdb_id = pdb_id.upper().strip()
    dest = cache_dir / f"{pdb_id}.pdb"
    shutil.copy2(source_path, dest)
    return str(dest)


def save_upload_to_user_cache(user_id: str, filename: str, file_bytes: bytes) -> str:
    """Save an uploaded file to the user's cache. Returns path."""
    cache_dir = get_user_cache_dir(user_id)
    base = Path(filename).stem
    suffix = Path(filename).suffix or ".pdb"
    dest = cache_dir / f"{base}{suffix}"
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return str(dest)


def list_user_cached(user_id: str) -> List[Tuple[str, str]]:
    """List (display_name, path) for all cached structures for the user."""
    cache_dir = get_user_cache_dir(user_id)
    if not cache_dir.exists():
        return []
    results = []
    for p in sorted(cache_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.suffix.lower() in (".pdb", ".cif"):
            results.append((p.stem, str(p)))
    return results


def load_from_user_cache(user_id: str, name: str) -> Optional[str]:
    """Load a cached structure by name. Returns path or None."""
    cache_dir = get_user_cache_dir(user_id)
    for suffix in (".pdb", ".cif"):
        p = cache_dir / f"{name}{suffix}"
        if p.exists():
            return str(p)
    return None
