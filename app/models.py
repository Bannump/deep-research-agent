"""SQLAlchemy ORM models for sessions, subqueries, and memory actions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    total_estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    summary_stats: Mapped[dict[str, Any]] = mapped_column(SQLITE_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    subqueries: Mapped[list["SubQueryRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    memory_actions: Mapped[list["MemoryAction"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SubQueryRecord(Base):
    __tablename__ = "subquery_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    subquery_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    episodic_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    retrieved_count: Mapped[int] = mapped_column(Integer, default=0)
    retained_count: Mapped[int] = mapped_column(Integer, default=0)
    discarded_count: Mapped[int] = mapped_column(Integer, default=0)
    working_memory_tokens: Mapped[int] = mapped_column(Integer, default=0)
    retained_evidence: Mapped[list[dict[str, Any]]] = mapped_column(SQLITE_JSON, default=list)
    discarded_evidence: Mapped[list[dict[str, Any]]] = mapped_column(SQLITE_JSON, default=list)

    session: Mapped["ResearchSession"] = relationship(back_populates="subqueries")


class MemoryAction(Base):
    __tablename__ = "memory_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False
    )
    step: Mapped[str] = mapped_column(String(64), nullable=False, default="pipeline")
    subquery_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    detail: Mapped[dict[str, Any]] = mapped_column(SQLITE_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["ResearchSession"] = relationship(back_populates="memory_actions")
