"""Tests for subquery sanitization and deterministic grounded summarization."""

from __future__ import annotations

from app.evidence_clean import clean_chunk_for_synthesis, clean_evidence_line
from app.grounded_synthesis import (
    grounded_episodic_summary,
    grounded_final_answer,
    grounded_subquery_answer,
    sanitize_subqueries,
    strip_subquery_prefixes,
    strip_user_facing_subquery_labels,
    subquery_topic_bucket,
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
    assert "[What" not in epi
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
    assert "AES-256" in fin or "TLS" in fin


def test_final_answer_honest_when_empty() -> None:
    fin = grounded_final_answer("What?", [], [])
    assert "empty" in fin.lower() or "pruned" in fin.lower()


def test_sanitize_collapses_double_question_marks() -> None:
    out = sanitize_subqueries(["What are the key items in the corpus??"], "fallback query here", max_n=3)
    assert out
    assert "??" not in out[0]


def test_evidence_clean_drops_heading_lines() -> None:
    assert clean_evidence_line("# Technical Controls") is None
    assert clean_evidence_line("## Recommendations") is None
    block = "# Technical Controls\n\nData at rest must use AES-256.\n\n## Risks\nStale policies persist."
    cleaned = clean_chunk_for_synthesis(block)
    assert "#" not in cleaned
    assert "AES-256" in cleaned
    assert "stale" in cleaned.lower()


def test_control_focused_final_leads_with_controls_not_risk_language() -> None:
    ev = [
        "[03_risks_and_recommendations.md]\n"
        "Stale policies: controls documented but not enforced in CI/CD.\n"
        "[02_technical_controls.md]\n"
        "Data at rest must use AES-256 or equivalent. "
        "Data in transit must use TLS 1.2+ with modern cipher suites.",
    ]
    fin = grounded_final_answer("What are the key controls in the sample corpus?", [], ev)
    assert "#" not in fin
    low = fin.lower()
    assert "aes" in low
    idx_aes = low.find("aes")
    idx_stale = low.find("stale")
    assert idx_aes != -1
    assert idx_stale == -1 or idx_aes < idx_stale
    assert "key controls" in low
    assert "tls" in low
    # Opening should be theme-level (multi-category), not a single narrow fact first.
    opener = low.split("\n")[0]
    assert "across the retained excerpts" in opener or "emphasizes" in opener or "strongest theme" in opener


def test_episodic_notes_stay_distinct_across_subqueries() -> None:
    ch = RetrievedChunk(
        chunk_id="1",
        source="mixed.md",
        text=(
            "Control: data at rest must use AES-256 or equivalent. "
            "Risk: stale policies are a common failure mode when enforcement is weak."
        ),
        score=0.9,
    )
    a1 = grounded_subquery_answer("What technical security controls are described?", [ch])
    a2 = grounded_subquery_answer("What risks, enforcement gaps, or recommendations are mentioned?", [ch])
    e1 = grounded_episodic_summary("What technical security controls are described?", a1, prior_episodic=[])
    e2 = grounded_episodic_summary(
        "What risks, enforcement gaps, or recommendations are mentioned?",
        a2,
        prior_episodic=[e1],
    )
    assert e1 != e2
    assert "aes" in e1.lower()
    assert "stale" in e2.lower() or "failure" in e2.lower()


def test_strip_user_facing_subquery_labels_removes_question_brackets() -> None:
    raw = "[What technical security controls are described?] : Centralized logging is mandatory."
    assert "[What" not in strip_user_facing_subquery_labels(raw)
    assert "Centralized logging" in strip_user_facing_subquery_labels(raw)


def test_no_bracket_subquery_labels_in_final_or_episodic() -> None:
    ev = ["[doc.md]\nCentralized logging is mandatory for all environments."]
    epi = [
        "[What technical security controls are described?] : Centralized logging is mandatory.",
    ]
    fin = grounded_final_answer("What are the key controls in the sample corpus?", epi, ev)
    assert "[What" not in fin
    assert fin.count("centralized logging") <= 1


def test_governance_subquery_prefers_governance_sentence_over_pure_crypto() -> None:
    ch = RetrievedChunk(
        chunk_id="1",
        source="mixed.md",
        text=(
            "Data at rest must use AES-256 or equivalent. "
            "Quarterly vendor security reviews are required for critical suppliers."
        ),
        score=0.9,
    )
    g = grounded_subquery_answer("What governance or review controls are described?", [ch])
    low = g.lower()
    assert "vendor" in low or "review" in low
    assert low.find("vendor") < low.find("aes") or "aes" not in low


def test_subquery_topic_bucket_classification() -> None:
    assert subquery_topic_bucket("What technical security controls are described?") == "technical"
    assert subquery_topic_bucket("What governance or review controls are described?") == "governance"
    assert subquery_topic_bucket("What risks, enforcement gaps, or recommendations are mentioned?") == "risk"
