from app.config import get_settings
from app.memory_manager import MemoryManager
from app.retriever import RetrievedChunk


def _chunk(cid: str, text: str, score: float, source: str = "doc.txt") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, source=source, text=text, score=score)


def test_pruning_respects_chunk_and_token_limits() -> None:
    get_settings.cache_clear()
    mm = MemoryManager()
    # Many overlapping chunks; duplicates should be discarded first
    base = "Repeated evidence text about compliance. " * 40
    chunks = [
        _chunk("1", base, 0.9),
        _chunk("2", base, 0.85),
        _chunk("3", "Different content " * 50, 0.8),
        _chunk("4", "More unique " * 50, 0.75),
        _chunk("5", "Extra " * 60, 0.7),
        _chunk("6", "Tail " * 60, 0.65),
        _chunk("7", "Low " * 20, 0.05),
        _chunk("8", "Also low " * 20, 0.04),
    ]
    res = mm.enforce_working_memory(chunks)
    assert len(res.retained) <= get_settings().max_retained_chunks
    assert res.estimated_tokens <= get_settings().working_memory_token_budget
    reasons = {d.reason for d in res.discarded}
    assert "duplicate" in reasons or "chunk_limit" in reasons or "memory_limit" in reasons
