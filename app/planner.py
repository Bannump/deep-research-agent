"""Query decomposition: up to 3 focused subqueries."""

from __future__ import annotations

import re
from typing import Any

from app.config import get_settings
from app.llm import LLMProvider, get_llm_provider


def _fallback_subqueries(original: str, max_n: int) -> list[str]:
    """Deterministic split when LLM output is unusable."""
    cleaned = original.strip()
    if not cleaned:
        return []
    # Split on numbered list or semicolons
    parts = re.split(r"(?:\n|^)\s*\d+[\).]\s+", cleaned)
    if len(parts) > 1:
        out = [p.strip() for p in parts if len(p.strip()) > 5]
        return out[:max_n]
    parts = [p.strip() for p in re.split(r"[;\n]+", cleaned) if len(p.strip()) > 10]
    if len(parts) >= 2:
        return parts[:max_n]
    # Single question → three angles
    base = cleaned[:200]
    return [
        f"What definitions and background apply to: {base}?",
        f"What mechanisms, processes, or constraints are described regarding: {base}?",
        f"What outcomes, risks, or recommendations are stated about: {base}?",
    ][:max_n]


def decompose_query(original_query: str, llm: LLMProvider | None = None) -> list[str]:
    """
    Produce up to MAX_SUBQUERIES focused subqueries.
    Prefers structured JSON from the LLM; falls back deterministically.
    """
    settings = get_settings()
    max_n = settings.max_subqueries
    llm = llm or get_llm_provider()

    system = (
        "You decompose research questions into focused subqueries. "
        "Return ONLY valid JSON: {\"subqueries\": [string, ...]} with at most "
        f"{max_n} items. Each subquery must be self-contained and specific."
    )
    user = f"Original question:\n{original_query}\n"

    data: dict[str, Any] = {}
    try:
        data = llm.complete_json(system, user, max_tokens=500)
    except Exception:
        data = {}

    raw_list = data.get("subqueries")
    subqueries: list[str] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str):
                s = item.strip()
                if len(s) > 5:
                    subqueries.append(s)

    subqueries = subqueries[:max_n]
    if len(subqueries) < 1:
        subqueries = _fallback_subqueries(original_query, max_n)
    return subqueries[:max_n]
