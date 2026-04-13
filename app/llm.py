"""LLM provider abstraction: pluggable backends + mock/demo mode."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import get_settings
from app.utils import normalize_ws


class LLMProvider(ABC):
    """Vendor-agnostic completion interface."""

    @abstractmethod
    def complete_text(self, system: str, user: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> dict[str, Any]:
        raw = self.complete_text(system, user, max_tokens=max_tokens)
        return _parse_json_loose(raw)


def _parse_json_loose(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class MockLLMProvider(LLMProvider):
    """
    Deterministic offline provider for demos and CI.
    Does not call external APIs.
    """

    def complete_text(self, system: str, user: str, max_tokens: int = 1024) -> str:
        _ = (system, max_tokens)
        u = user.lower()
        if "subqueries" in system.lower() or "decompose" in system.lower():
            parts = re.split(r"[;\n]+", user)
            parts = [normalize_ws(p) for p in parts if len(normalize_ws(p)) > 10][:3]
            if len(parts) < 2:
                parts = [
                    f"What background context applies: {user[:80]}?",
                    f"What mechanisms or constraints are described: {user[:80]}?",
                    f"What conclusions or recommendations appear: {user[:80]}?",
                ]
            return json.dumps({"subqueries": parts[:3]})
        # Subquery answer
        return (
            "[Mock answer] Based on the retained evidence snippets, the corpus discusses "
            "related themes; specifics depend on source filenames cited in context."
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> dict[str, Any]:
        txt = self.complete_text(system, user, max_tokens=max_tokens)
        return _parse_json_loose(txt)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions API (https://api.openai.com/v1/chat/completions)."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._url = "https://api.openai.com/v1/chat/completions"

    def complete_text(self, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120.0) as client:
            r = client.post(self._url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"] or ""


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._url = "https://api.anthropic.com/v1/messages"

    def complete_text(self, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120.0) as client:
            r = client.post(self._url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        parts = data.get("content") or []
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        return "".join(texts)


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAICompatibleProvider(settings.openai_api_key, settings.openai_model)
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    return MockLLMProvider()
