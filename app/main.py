"""FastAPI application: constrained research pipeline and session APIs."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.config import get_settings
from app.evidence_clean import clean_chunk_for_synthesis
from app.db import get_db_session, get_session_factory, init_db
from app.evaluator import (
    build_pipeline_demonstration,
    build_reviewer_note,
    build_session_metrics,
    count_text_tokens,
    rollup_session_tokens,
)
from app.llm import get_llm_provider
from app.memory_manager import MemoryManager
from app.models import MemoryAction as MemoryActionORM
from app.models import ResearchSession as ResearchSessionORM
from app.models import SubQueryRecord
from app.planner import decompose_query
from app.retriever import get_retriever
from app.schemas import (
    DiscardRecord,
    MemoryActionOut,
    PipelineDemonstration,
    ResearchRequest,
    ResearchResponse,
    RetainedEvidenceSnippet,
    RuntimeConstraints,
    SessionDetailResponse,
    SessionListResponse,
    SessionMetrics,
    SessionSummary,
    SubQueryResultOut,
)
from app.synthesizer import answer_subquery, synthesize_final_answer
from app.utils import new_session_id

logger = logging.getLogger("app.research")


def _preview(text: str, n: int = 120) -> str:
    t = " ".join(text.split())
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _configure_logging()
    init_db()
    get_settings().data_dir.mkdir(parents=True, exist_ok=True)
    get_retriever()
    yield


app = FastAPI(
    title="Deep Research Agent",
    version=__version__,
    lifespan=_lifespan,
)


def get_db() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _run_pipeline(original_query: str) -> ResearchResponse:
    settings = get_settings()
    llm = get_llm_provider()
    memory = MemoryManager()
    retriever = get_retriever()

    session_id = new_session_id()
    memory_actions: list[MemoryActionOut] = []

    def log_action(
        step: str,
        action: str,
        subquery_index: int | None = None,
        detail: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        memory_actions.append(
            MemoryActionOut(
                step=step,
                subquery_index=subquery_index,
                action=action,
                detail=detail,
                meta=meta or {},
            )
        )

    constraints = RuntimeConstraints(
        max_subqueries=settings.max_subqueries,
        max_retrieved_chunks_per_subquery=settings.max_retrieved_chunks,
        max_retained_chunks_working_memory=settings.max_retained_chunks,
        working_memory_token_budget_estimated=settings.working_memory_token_budget,
    )

    planner_sys = "You decompose research questions into focused subqueries."
    planner_user = f"Original question:\n{original_query}\n"
    planner_tokens = count_text_tokens(planner_sys, planner_user)

    subqueries = decompose_query(original_query, llm=llm)
    log_action(
        "plan",
        "decompose",
        detail=f"Generated {len(subqueries)} subqueries.",
        meta={"subqueries": subqueries},
    )

    episodic_notes: list[str] = []
    subquery_results: list[SubQueryResultOut] = []
    subquery_prompt_token_list: list[int] = []
    total_retrieved = 0
    total_retained = 0
    total_discarded = 0
    reason_counter: Counter[str] = Counter()
    evidence_for_final: list[str] = []

    for idx, sq in enumerate(subqueries):
        chunks = retriever.retrieve(sq, top_k=settings.max_retrieved_chunks)
        total_retrieved += len(chunks)
        log_action(
            "retrieve",
            "chroma_query",
            subquery_index=idx,
            detail=f"Retrieved {len(chunks)} candidate chunks (cap {settings.max_retrieved_chunks}).",
            meta={"subquery": sq},
        )

        wm = memory.enforce_working_memory(chunks)
        total_retained += len(wm.retained)
        total_discarded += len(wm.discarded)
        for d in wm.discarded:
            reason_counter[d.reason] += 1

        for d in wm.discarded:
            log_action(
                "prune",
                "discard_chunk",
                subquery_index=idx,
                detail=d.detail,
                meta={"chunk_id": d.chunk_id, "source": d.source, "reason": d.reason},
            )

        log_action(
            "working_memory",
            "enforce",
            subquery_index=idx,
            detail=(
                f"Retained {len(wm.retained)} chunks, ~{wm.estimated_tokens} tokens "
                f"(budget {settings.working_memory_token_budget})."
            ),
        )

        sub_ans = answer_subquery(sq, wm.retained, llm=llm)
        sq_tokens = count_text_tokens(
            "system prompt",
            sq,
            "\n".join(c.text for c in wm.retained),
        )
        subquery_prompt_token_list.append(sq_tokens)

        epi = memory.summarize_episodic(sq, sub_ans, llm=llm, prior_episodic=list(episodic_notes))
        episodic_notes.append(epi)
        for c in wm.retained[:4]:
            cleaned = clean_chunk_for_synthesis(c.text) or c.text
            evidence_for_final.append(f"[{c.source}]\n{cleaned[:800]}")

        log_action(
            "episodic",
            "summarize",
            subquery_index=idx,
            detail="Compressed subquery answer into episodic note for synthesis.",
        )

        discarded_records = [
            DiscardRecord(
                chunk_id=d.chunk_id,
                source=d.source,
                reason=d.reason,
                detail=d.detail,
            )
            for d in wm.discarded
        ]
        retained_snippets = [
            RetainedEvidenceSnippet(
                chunk_id=c.chunk_id,
                source=c.source,
                text_preview=_preview(c.text),
            )
            for c in wm.retained
        ]

        subquery_results.append(
            SubQueryResultOut(
                subquery_index=idx,
                question=sq,
                retrieved_count=len(chunks),
                retained_count=len(wm.retained),
                discarded_count=len(wm.discarded),
                retained_sources=sorted({c.source for c in wm.retained}),
                retained_snippets=retained_snippets,
                discarded_chunks=discarded_records,
                subquery_answer=sub_ans,
                episodic_summary=epi,
                working_memory_tokens=wm.estimated_tokens,
            )
        )

    synth_tokens = count_text_tokens(
        original_query,
        "\n".join(episodic_notes),
        "\n".join(evidence_for_final),
    )
    final_answer = synthesize_final_answer(
        original_query,
        episodic_notes,
        evidence_for_final,
        llm=llm,
    )

    rollup = rollup_session_tokens(
        planner_tokens=planner_tokens,
        subquery_prompt_tokens=subquery_prompt_token_list,
        synthesis_tokens=synth_tokens,
    )

    session_metrics = build_session_metrics(
        subquery_count=len(subqueries),
        total_retrieved=total_retrieved,
        total_retained=total_retained,
        total_discarded=total_discarded,
        discard_breakdown=dict(reason_counter),
        estimated_tokens_total=rollup.total_tokens,
        estimated_cost_usd=rollup.estimated_cost_usd,
    )

    demonstration = build_pipeline_demonstration(
        total_retrieved=total_retrieved,
        total_retained=total_retained,
        total_discarded=total_discarded,
        discard_breakdown=reason_counter,
        episodic_summaries=episodic_notes,
        evidence_excerpt_count=len(evidence_for_final),
    )

    reviewer_note = build_reviewer_note(
        session_id=session_id,
        metrics=session_metrics,
        episodic_count=len(episodic_notes),
    )

    logger.info(
        "Research session %s | subqueries=%d | chunks ret/kept/drop=%d/%d/%d | ~%d tokens",
        session_id,
        len(subqueries),
        total_retrieved,
        total_retained,
        total_discarded,
        rollup.total_tokens,
    )

    out = ResearchResponse(
        session_id=session_id,
        original_query=original_query,
        constraints=constraints,
        subqueries=subqueries,
        subquery_results=subquery_results,
        episodic_summaries=episodic_notes,
        final_answer=final_answer,
        total_estimated_tokens=rollup.total_tokens,
        total_estimated_cost=rollup.estimated_cost_usd,
        session_metrics=session_metrics,
        demonstration=demonstration,
        reviewer_note=reviewer_note,
        memory_actions=memory_actions,
    )

    _persist_session(out)
    return out


def _pack_summary_stats(response: ResearchResponse) -> dict[str, Any]:
    return {
        "version": 2,
        "constraints": response.constraints.model_dump(),
        "session_metrics": response.session_metrics.model_dump(),
        "demonstration": response.demonstration.model_dump(),
        "reviewer_note": response.reviewer_note,
        "episodic_summaries": response.episodic_summaries,
        "subqueries": response.subqueries,
    }


def _persist_session(response: ResearchResponse) -> None:
    packed = _pack_summary_stats(response)
    with get_db_session() as db:
        rs = ResearchSessionORM(
            session_id=response.session_id,
            original_query=response.original_query,
            final_answer=response.final_answer,
            total_estimated_tokens=response.total_estimated_tokens,
            total_estimated_cost_usd=response.total_estimated_cost,
            summary_stats=packed,
        )
        db.add(rs)
        db.flush()

        for i, pq in enumerate(response.subquery_results):
            retained_evidence = [s.model_dump() for s in pq.retained_snippets]
            discarded_evidence = [d.model_dump() for d in pq.discarded_chunks]
            db.add(
                SubQueryRecord(
                    session_id=rs.id,
                    order_index=i,
                    question=pq.question,
                    subquery_answer=pq.subquery_answer,
                    episodic_summary=pq.episodic_summary,
                    retrieved_count=pq.retrieved_count,
                    retained_count=pq.retained_count,
                    discarded_count=pq.discarded_count,
                    working_memory_tokens=pq.working_memory_tokens,
                    retained_evidence=retained_evidence,
                    discarded_evidence=discarded_evidence,
                )
            )

        for m in response.memory_actions:
            db.add(
                MemoryActionORM(
                    session_id=rs.id,
                    step=m.step,
                    subquery_index=m.subquery_index,
                    action_type=m.action,
                    message=m.detail,
                    detail=m.meta,
                )
            )


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    q = req.query.strip()
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short.")
    return _run_pipeline(q)


def _snippets_from_db(rows: list[dict[str, Any]]) -> list[RetainedEvidenceSnippet]:
    out: list[RetainedEvidenceSnippet] = []
    for row in rows:
        if "text_preview" in row and "chunk_id" in row:
            out.append(RetainedEvidenceSnippet.model_validate(row))
        elif row.get("source"):
            out.append(
                RetainedEvidenceSnippet(
                    chunk_id=row.get("chunk_id", ""),
                    source=str(row["source"]),
                    text_preview=row.get("text_preview") or "",
                )
            )
    return out


def _discards_from_db(rows: list[dict[str, Any]]) -> list[DiscardRecord]:
    out: list[DiscardRecord] = []
    for row in rows:
        if not row:
            continue
        try:
            out.append(DiscardRecord.model_validate(row))
        except Exception:
            continue
    return out


def _orm_to_detail(rs: ResearchSessionORM) -> SessionDetailResponse:
    subqs = sorted(rs.subqueries, key=lambda x: x.order_index)
    stats = rs.summary_stats or {}

    subquery_results: list[SubQueryResultOut] = []
    for s in subqs:
        retained_raw = list(s.retained_evidence or [])
        discarded_raw = list(s.discarded_evidence or [])
        snippets = _snippets_from_db(retained_raw) if retained_raw else []
        if not snippets and retained_raw:
            snippets = [
                RetainedEvidenceSnippet(
                    chunk_id=str(x.get("chunk_id", "")),
                    source=str(x.get("source", "")),
                    text_preview=str(x.get("text_preview", "")),
                )
                for x in retained_raw
                if isinstance(x, dict)
            ]
        discards = _discards_from_db(discarded_raw) if discarded_raw else []
        subquery_results.append(
            SubQueryResultOut(
                subquery_index=s.order_index,
                question=s.question,
                retrieved_count=s.retrieved_count,
                retained_count=s.retained_count,
                discarded_count=s.discarded_count,
                retained_sources=sorted({e.get("source", "") for e in retained_raw if isinstance(e, dict)}),
                retained_snippets=snippets,
                discarded_chunks=discards,
                subquery_answer=s.subquery_answer,
                episodic_summary=s.episodic_summary,
                working_memory_tokens=s.working_memory_tokens,
            )
        )

    mem_out = [
        MemoryActionOut(
            step=m.step,
            subquery_index=m.subquery_index,
            action=m.action_type,
            detail=m.message,
            meta=m.detail if isinstance(m.detail, dict) else {},
        )
        for m in sorted(rs.memory_actions, key=lambda x: x.id)
    ]

    constraints = RuntimeConstraints.model_validate(stats["constraints"]) if stats.get("constraints") else None
    session_metrics = SessionMetrics.model_validate(stats["session_metrics"]) if stats.get("session_metrics") else None
    demonstration = (
        PipelineDemonstration.model_validate(stats["demonstration"]) if stats.get("demonstration") else None
    )
    reviewer_note = stats.get("reviewer_note")
    episodic_from_stats = stats.get("episodic_summaries")

    return SessionDetailResponse(
        session_id=rs.session_id,
        original_query=rs.original_query,
        constraints=constraints,
        subqueries=[s.question for s in subqs],
        subquery_results=subquery_results,
        episodic_summaries=list(episodic_from_stats or [s.episodic_summary for s in subqs]),
        final_answer=rs.final_answer,
        total_estimated_tokens=rs.total_estimated_tokens,
        total_estimated_cost=rs.total_estimated_cost_usd,
        session_metrics=session_metrics,
        demonstration=demonstration,
        reviewer_note=reviewer_note,
        memory_actions=mem_out,
        summary_stats=stats,
        created_at=rs.created_at.isoformat() if rs.created_at else None,
    )


@app.get("/session/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionDetailResponse:
    rs = db.scalars(select(ResearchSessionORM).where(ResearchSessionORM.session_id == session_id)).first()
    if rs is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _orm_to_detail(rs)


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions(limit: int = 20, db: Session = Depends(get_db)) -> SessionListResponse:
    limit = min(max(limit, 1), 100)
    rows = db.scalars(select(ResearchSessionORM).order_by(ResearchSessionORM.id.desc()).limit(limit)).all()
    sessions: list[SessionSummary] = []
    for r in rows:
        stats = r.summary_stats or {}
        sm = stats.get("session_metrics") or {}
        subq_count = sm.get("subquery_count")
        if subq_count is None:
            subq_count = stats.get("num_subqueries")
        preview = r.original_query[:120] + ("…" if len(r.original_query) > 120 else "")
        cost = float(r.total_estimated_cost_usd)
        one_line = f"{subq_count or '?'} subqueries · ~{r.total_estimated_tokens} tok · ${cost:.4f}"
        sessions.append(
            SessionSummary(
                session_id=r.session_id,
                query_preview=preview,
                subquery_count=int(subq_count) if subq_count is not None else None,
                created_at=r.created_at.isoformat() if r.created_at else None,
                total_estimated_tokens=r.total_estimated_tokens,
                total_estimated_cost=cost,
                one_line=one_line,
            )
        )
    return SessionListResponse(sessions=sessions)
