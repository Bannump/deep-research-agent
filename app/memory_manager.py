"""Working-memory enforcement, pruning, discard logging, episodic summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import get_settings
from app.grounded_synthesis import grounded_episodic_summary
from app.llm import LLMProvider, MockLLMProvider, get_llm_provider
from app.retriever import RetrievedChunk
from app.utils import chunk_text_similarity_key, estimate_tokens, normalize_ws

DiscardReason = Literal["low_relevance", "duplicate", "memory_limit", "chunk_limit"]


@dataclass
class DiscardedChunk:
    chunk_id: str
    source: str
    reason: DiscardReason
    detail: str


@dataclass
class WorkingMemoryResult:
    retained: list[RetrievedChunk]
    discarded: list[DiscardedChunk]
    estimated_tokens: int


class MemoryManager:
    """
    Enforces max retained chunks and approximate token budget per subquery step.
    Exposes why each chunk was not kept.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def enforce_working_memory(self, chunks: list[RetrievedChunk]) -> WorkingMemoryResult:
        max_chunks = self._settings.max_retained_chunks
        budget = self._settings.working_memory_token_budget

        # Sort by relevance score (desc)
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        top_score = sorted_chunks[0].score if sorted_chunks else 0.0
        low_threshold = max(0.08, 0.25 * float(top_score))

        retained: list[RetrievedChunk] = []
        discarded: list[DiscardedChunk] = []
        seen_keys: set[str] = set()

        # Pass 1: mark duplicates (relative to higher-scoring seen content)
        deduped: list[RetrievedChunk] = []
        for c in sorted_chunks:
            key = chunk_text_similarity_key(c.text)
            if key in seen_keys:
                discarded.append(
                    DiscardedChunk(
                        chunk_id=c.chunk_id,
                        source=c.source,
                        reason="duplicate",
                        detail="Near-duplicate of a higher-ranked chunk.",
                    )
                )
                continue
            seen_keys.add(key)
            deduped.append(c)

        # Pass 2: greedy fill by score with token budget
        token_sum = 0
        for c in deduped:
            t = estimate_tokens(c.text)
            if c.score < low_threshold:
                discarded.append(
                    DiscardedChunk(
                        chunk_id=c.chunk_id,
                        source=c.source,
                        reason="low_relevance",
                        detail=f"Score {c.score:.3f} below relative threshold {low_threshold:.3f}.",
                    )
                )
                continue
            if len(retained) >= max_chunks:
                discarded.append(
                    DiscardedChunk(
                        chunk_id=c.chunk_id,
                        source=c.source,
                        reason="chunk_limit",
                        detail=f"Working memory allows at most {max_chunks} chunks.",
                    )
                )
                continue
            if token_sum + t > budget:
                # Try to still include if we're under chunk limit — budget blocks
                discarded.append(
                    DiscardedChunk(
                        chunk_id=c.chunk_id,
                        source=c.source,
                        reason="memory_limit",
                        detail=f"Adding ~{t} tokens would exceed ~{budget} token budget (running {token_sum}).",
                    )
                )
                continue
            retained.append(c)
            token_sum += t

        return WorkingMemoryResult(
            retained=retained,
            discarded=discarded,
            estimated_tokens=token_sum,
        )

    def summarize_episodic(
        self,
        subquery: str,
        answer: str,
        llm: LLMProvider | None = None,
    ) -> str:
        """Compress subquery outcome into a short episodic note for SQLite."""
        llm = llm or get_llm_provider()
        if isinstance(llm, MockLLMProvider):
            return grounded_episodic_summary(subquery, answer)

        system = (
            "Summarize the subquery result in 1-2 concise sentences for episodic memory. "
            "Shorter than the full answer. No bullet points. "
            "Keep concrete nouns (products, standards, control types). Do not add new facts."
        )
        user = f"Subquery: {subquery}\nAnswer:\n{answer[:4000]}\n"
        try:
            out = normalize_ws(llm.complete_text(system, user, max_tokens=256))
            return out[:1200]
        except Exception:
            return grounded_episodic_summary(subquery, answer)
