"""Query decomposition: up to 3 focused subqueries."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.evidence_clean import collapse_repeated_punctuation
from app.grounded_synthesis import (
    intent_aware_fallback_subqueries,
    is_cleaned_original_passthrough,
    query_is_control_focused,
    sanitize_subqueries,
    strip_subquery_prefixes,
    subqueries_quality_poor,
)
from app.llm import LLMProvider, get_llm_provider


def decompose_query(original_query: str, llm: LLMProvider | None = None) -> list[str]:
    """
    Produce up to MAX_SUBQUERIES focused subqueries.
    Prefers structured JSON from the LLM; falls back deterministically.
    """
    settings = get_settings()
    max_n = settings.max_subqueries
    llm = llm or get_llm_provider()

    system = (
        "You decompose research questions into short, natural subqueries for retrieval. "
        "Each subquery must be a standalone question or focused search phrase (not a meta wrapper). "
        "Avoid generic templates like 'What definitions and background apply to...'. "
        "Prefer concrete angles (technical controls, governance, risks, requirements). "
        f"Return ONLY valid JSON: {{\"subqueries\": [string, ...]}} with at most {max_n} items."
    )
    user = f"Original question:\n{original_query}\n"

    data: dict[str, Any] = {}
    try:
        data = llm.complete_json(system, user, max_tokens=500)
    except Exception:
        data = {}

    raw_list = data.get("subqueries")
    raw_subqueries: list[str] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str):
                raw_subqueries.append(item)

    cleaned_original = collapse_repeated_punctuation(strip_subquery_prefixes(original_query))

    subqueries = sanitize_subqueries(raw_subqueries, original_query, max_n)

    if len(subqueries) < 1:
        subqueries = intent_aware_fallback_subqueries(cleaned_original, max_n)
        subqueries = sanitize_subqueries(subqueries, cleaned_original, max_n)
    elif subqueries_quality_poor(subqueries, original_query):
        subqueries = intent_aware_fallback_subqueries(cleaned_original, max_n)
        subqueries = sanitize_subqueries(subqueries, cleaned_original, max_n)
    elif (
        query_is_control_focused(original_query)
        and is_cleaned_original_passthrough(subqueries, cleaned_original)
    ):
        subqueries = intent_aware_fallback_subqueries(cleaned_original, max_n)
        subqueries = sanitize_subqueries(subqueries, cleaned_original, max_n)

    if len(subqueries) < 1 and cleaned_original and len(cleaned_original) >= 8:
        subqueries = [cleaned_original][:max_n]

    return subqueries[:max_n]
