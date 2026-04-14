from app.config import get_settings
from app.llm import LLMProvider, MockLLMProvider
from app.planner import decompose_query


def test_decompose_max_three() -> None:
    get_settings.cache_clear()
    subs = decompose_query(
        "Explain Aurora compliance; compare encryption controls; summarize risks.",
        llm=MockLLMProvider(),
    )
    assert 1 <= len(subs) <= get_settings().max_subqueries


class _JunkSubqueryLLM(LLMProvider):
    """Simulates an LLM that echoes label junk alongside one real subquery."""

    def complete_text(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return ""

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        return {
            "subqueries": [
                "Original question:",
                "Question:",
                "What penetration testing evidence does the checklist require?",
            ]
        }


def test_decompose_sanitizes_junk_subqueries() -> None:
    get_settings.cache_clear()
    subs = decompose_query(
        "According to the sample corpus, what does the security checklist require for pentesting?",
        llm=_JunkSubqueryLLM(),
    )
    assert len(subs) >= 1
    assert all("original question" not in s.lower().strip() for s in subs)
    assert all(s.strip() for s in subs)
    assert len(subs) <= get_settings().max_subqueries
