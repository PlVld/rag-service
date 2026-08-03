from __future__ import annotations

from pathlib import Path
from typing import Dict

_CACHE: Dict[str, str] = {}


def load_openapi_md(filename: str) -> str:
    """
    Loads endpoint descriptions from app/openapi_md/*.md.
    Returns empty string if file missing/unreadable.
    """
    cached = _CACHE.get(filename)
    if cached is not None:
        return cached

    base_dir = Path(__file__).resolve().parent
    md_path = base_dir / filename
    try:
        text = md_path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError):
        text = ""
    _CACHE[filename] = text
    return text

