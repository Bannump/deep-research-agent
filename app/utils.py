"""Shared helpers: IDs, timestamps, token estimation."""

from __future__ import annotations

import re
import uuid


def new_session_id() -> str:
    return str(uuid.uuid4())


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Rough token estimate: ~4 characters per token for English prose.
    Not aligned with any specific tokenizer; documented as approximation.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text_similarity_key(text: str) -> str:
    """Normalize for duplicate detection."""
    return normalize_ws(text.lower())[:500]

