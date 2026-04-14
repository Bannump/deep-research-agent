"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Multi-part research question")


DiscardReason = Literal["low_relevance", "duplicate", "memory_limit", "chunk_limit"]


class DiscardRecord(BaseModel):
    """One chunk removed during working-memory enforcement, with an explicit reason."""

    chunk_id: str
    source: str
    reason: DiscardReason
    detail: str = ""


class RetainedEvidenceSnippet(BaseModel):
    """Short preview of evidence kept in working memory for this subquery."""

    chunk_id: str
    source: str
    text_preview: str = Field(..., description="First ~120 characters of retained chunk text")


class RuntimeConstraints(BaseModel):
    """Active pipeline limits for this run (for reviewer transparency)."""

    max_subqueries: int
    max_retrieved_chunks_per_subquery: int
    max_retained_chunks_working_memory: int
    working_memory_token_budget_estimated: int


class SessionMetrics(BaseModel):
    """Roll-up statistics for the session."""

    subquery_count: int
    chunks_retrieved_total: int
    chunks_retained_total: int
    chunks_discarded_total: int
    retention_rate: float = Field(description="retained / retrieved across all subquery steps")
    discard_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Counts by DiscardRecord.reason across the session",
    )
    estimated_tokens_total: int
    estimated_cost_usd: float


class PipelineDemonstration(BaseModel):
    """
    Plain-language pointers for reviewers: how pruning, episodic memory, and synthesis show up
    in this response (complements structured fields).
    """

    memory_pruning: str
    episodic_compression: str
    final_synthesis: str


class SubQueryResultOut(BaseModel):
    """One decomposition step: retrieval → working memory → answer → episodic note."""

    subquery_index: int = Field(..., ge=0, description="0-based order in this session")
    question: str
    retrieved_count: int
    retained_count: int
    discarded_count: int
    retained_sources: list[str] = Field(description="Distinct source filenames for retained chunks")
    retained_snippets: list[RetainedEvidenceSnippet] = Field(
        default_factory=list,
        description="What actually entered working memory (preview only)",
    )
    discarded_chunks: list[DiscardRecord] = Field(
        default_factory=list,
        description="Chunks not passed to the subquery LLM, with reasons",
    )
    subquery_answer: str
    episodic_summary: str = Field(description="Compressed note stored for later synthesis (episodic tier)")
    working_memory_tokens: int = Field(description="Estimated tokens for retained chunk text in this step")


class MemoryActionOut(BaseModel):
    step: str
    subquery_index: int | None = None
    action: str
    detail: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    """
    Full session output. Reviewers typically read `reviewer_note`, `demonstration`,
    `subquery_results` (pruning + episodic per step), then `final_answer`.
    """

    session_id: str
    original_query: str
    constraints: RuntimeConstraints
    subqueries: list[str]
    subquery_results: list[SubQueryResultOut]
    episodic_summaries: list[str] = Field(
        default_factory=list,
        description="Ordered episodic notes (one per subquery) fed forward toward synthesis",
    )
    final_answer: str
    total_estimated_tokens: int = Field(description="Same as session_metrics.estimated_tokens_total (quick scan)")
    total_estimated_cost: float = Field(description="Same as session_metrics.estimated_cost_usd (quick scan)")
    session_metrics: SessionMetrics
    demonstration: PipelineDemonstration
    reviewer_note: str = Field(
        description="One-paragraph guide to interpreting this JSON for assessment/review",
    )
    memory_actions: list[MemoryActionOut]


class SessionDetailResponse(BaseModel):
    """Persisted session (same shape as ResearchResponse for API consistency)."""

    session_id: str
    original_query: str
    constraints: RuntimeConstraints | None = None
    subqueries: list[str]
    subquery_results: list[SubQueryResultOut]
    episodic_summaries: list[str] = Field(default_factory=list)
    final_answer: str
    total_estimated_tokens: int
    total_estimated_cost: float
    session_metrics: SessionMetrics | None = None
    demonstration: PipelineDemonstration | None = None
    reviewer_note: str | None = None
    memory_actions: list[MemoryActionOut]
    summary_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy blob from DB; prefer session_metrics when present",
    )
    created_at: str | None = None


class SessionSummary(BaseModel):
    """Row for GET /sessions — scannable overview of recent runs."""

    session_id: str
    query_preview: str = Field(description="Truncated original question")
    subquery_count: int | None = None
    created_at: str | None
    total_estimated_tokens: int
    total_estimated_cost: float
    one_line: str = Field(
        default="",
        description="Human-readable summary for logs and UI lists",
    )


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
