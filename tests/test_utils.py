from app.utils import chunk_text_similarity_key, estimate_tokens, normalize_ws


def test_estimate_tokens_positive() -> None:
    assert estimate_tokens("a" * 400) == 100


def test_normalize_ws() -> None:
    assert normalize_ws("  hello   world  ") == "hello world"


def test_similarity_key_stable() -> None:
    a = chunk_text_similarity_key("Hello  WORLD")
    b = chunk_text_similarity_key("hello world")
    assert a == b
