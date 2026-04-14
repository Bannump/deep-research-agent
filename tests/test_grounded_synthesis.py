"""Tests for subquery sanitization and deterministic grounded summarization."""

from __future__ import annotations

from app.grounded_synthesis import (
    grounded_episodic_summary,
    grounded_final_answer,
    grounded_subquery_answer,
    sanitize_subqueries,
    strip_subquery_prefixes,
)
from app.retriever import RetrievedChunk


def test_strip_original_question_prefix() -> None:
    assert strip_subquery_prefixes("Original question: What is TLS?") == "What is TLS?"
    assert strip_subquery_prefixes("Question:  AES-256") == "AES-256"


def test_sanitize_drops_junk_and_dedupes() -> None:
    raw = [
        "Original question:",
        "Question:",
        "",
        "What are encryption requirements?",
        "what are encryption requirements?",  # duplicate when normalized
        "short",
    ]
    out = sanitize_subqueries(raw, "fallback original query text here", max_n=3)
    assert len(out) == 1
    assert "encryption" in out[0].lower()


def test_sanitize_falls_back_to_cleaned_original() -> None:
    out = sanitize_subqueries([], "What are the key controls in the sample corpus?", max_n=3)
    assert len(out) == 1
    assert "controls" in out[0].lower()


def test_grounded_subquery_lists_facts_with_sources() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="1",
            source="02_technical_controls.md",
            text=(
                "Data at rest must use AES-256. Data in transit must use TLS 1.2+ "
                "with modern cipher suites."
            ),
            score=0.9,
        )
    ]
    ans = grounded_subquery_answer("What encryption is required?", chunks)
    assert "[Mock answer]" not in ans
    assert "AES-256" in ans
    assert "TLS" in ans
    assert "02_technical_controls.md" in ans


def test_episodic_shorter_and_distinct_from_subquery() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="1",
            source="doc.md",
            text="Vendor security review is required annually for critical suppliers.",
            score=0.8,
        )
    ]
    sub = grounded_subquery_answer("Vendor requirements?", chunks)
    epi = grounded_episodic_summary("Vendor requirements?", sub)
    assert len(epi) < len(sub)
    assert epi.startswith("[")
    assert "Vendor security review" in epi
    assert sub not in epi


def test_final_answer_addresses_query_with_concrete_terms() -> None:
    ev = [
        "[02_technical_controls.md]\n"
        "AES-256 at rest; TLS 1.2+ in transit; quarterly key rotation in KMS."
    ]
    episodic = [
        "Episodic note (encryption): retained text specifies AES-256 and TLS 1.2+."
    ]
    fin = grounded_final_answer("What encryption standards apply?", episodic, ev)
    assert "[Mock answer]" not in fin
    assert "Direct answer" in fin or "AES" in fin
    assert "AES-256" in fin or "TLS" in fin


def test_final_answer_honest_when_empty() -> None:
    fin = grounded_final_answer("What?", [], [])
    assert "empty" in fin.lower() or "pruned" in fin.lower()
