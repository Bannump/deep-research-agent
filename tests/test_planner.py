from app.config import get_settings
from app.llm import MockLLMProvider
from app.planner import decompose_query


def test_decompose_max_three() -> None:
    get_settings.cache_clear()
    subs = decompose_query(
        "Explain Aurora compliance; compare encryption controls; summarize risks.",
        llm=MockLLMProvider(),
    )
    assert 1 <= len(subs) <= get_settings().max_subqueries
