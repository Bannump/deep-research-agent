"""Session-level token/cost estimates, metrics, and reviewer-facing copy."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.config import get_settings
from app.schemas import PipelineDemonstration, SessionMetrics
from app.utils import estimate_tokens


@dataclass
class TokenCostEstimate:
    total_tokens: int
    estimated_cost_usd: float


def estimate_cost_usd(total_tokens: int) -> float:
    settings = get_settings()
    return (total_tokens / 1000.0) * settings.default_cost_per_1k_tokens_usd


def rollup_session_tokens(
    planner_tokens: int,
    subquery_prompt_tokens: list[int],
    synthesis_tokens: int,
) -> TokenCostEstimate:
    total = planner_tokens + sum(subquery_prompt_tokens) + synthesis_tokens
    return TokenCostEstimate(total_tokens=total, estimated_cost_usd=estimate_cost_usd(total))


def count_text_tokens(*parts: str) -> int:
    return sum(estimate_tokens(p) for p in parts)


def merge_discard_reason_counts(breakdown: dict[str, int], reasons: list[str]) -> None:
    for r in reasons:
        breakdown[r] = breakdown.get(r, 0) + 1


def build_session_metrics(
    *,
    subquery_count: int,
    total_retrieved: int,
    total_retained: int,
    total_discarded: int,
    discard_breakdown: dict[str, int],
    estimated_tokens_total: int,
    estimated_cost_usd: float,
) -> SessionMetrics:
    return SessionMetrics(
        subquery_count=subquery_count,
        chunks_retrieved_total=total_retrieved,
        chunks_retained_total=total_retained,
        chunks_discarded_total=total_discarded,
        retention_rate=(total_retained / total_retrieved) if total_retrieved else 0.0,
        discard_breakdown=dict(sorted(discard_breakdown.items())),
        estimated_tokens_total=estimated_tokens_total,
        estimated_cost_usd=estimated_cost_usd,
    )


def build_pipeline_demonstration(
    *,
    total_retrieved: int,
    total_retained: int,
    total_discarded: int,
    discard_breakdown: Counter[str],
    episodic_summaries: list[str],
    evidence_excerpt_count: int,
) -> PipelineDemonstration:
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(discard_breakdown.items()) if v)
    return PipelineDemonstration(
        memory_pruning=(
            f"Retrieval produced {total_retrieved} candidate chunks across subqueries; working memory retained "
            f"{total_retained} and discarded {total_discarded}. Reason counts: {reasons or 'n/a'}. "
            "Ordering is by relevance score, then deduplication, then strict chunk/token caps."
        ),
        episodic_compression=(
            f"Each subquery answer was compressed into {len(episodic_summaries)} episodic note(s) "
            "(see `episodic_summaries`). That tier is intentionally shorter than raw evidence and is what "
            "cross-step synthesis relies on besides short excerpts."
        ),
        final_synthesis=(
            f"The final answer was generated from those episodic notes plus {evidence_excerpt_count} "
            "short retained-chunk excerpt(s)—not from the full retrieved set."
        ),
    )


def build_reviewer_note(
    *,
    session_id: str,
    metrics: SessionMetrics,
    episodic_count: int,
) -> str:
    return (
        f"Session `{session_id[:8]}…`: {metrics.subquery_count} subqueries. "
        f"Pruning: {metrics.chunks_retained_total}/{metrics.chunks_retrieved_total} chunks kept after working-memory rules; "
        f"{metrics.chunks_discarded_total} dropped (see `subquery_results[].discarded_chunks`). "
        f"Episodic tier: {episodic_count} note(s) in `episodic_summaries` → `final_answer`. "
        f"~{metrics.estimated_tokens_total} est. tokens; ~${metrics.estimated_cost_usd:.4f} under the configured cost model."
    )
