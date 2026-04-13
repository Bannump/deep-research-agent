"""Final synthesis from episodic summaries + selected evidence."""

from __future__ import annotations

from app.llm import LLMProvider, get_llm_provider
from app.retriever import RetrievedChunk
from app.utils import normalize_ws


def answer_subquery(
    subquery: str,
    retained: list[RetrievedChunk],
    llm: LLMProvider | None = None,
) -> str:
    """Generate an answer for one subquery using only retained evidence."""
    llm = llm or get_llm_provider()
    ctx = "\n\n".join(
        f"[{c.source}] (chunk {c.chunk_id})\n{c.text}" for c in retained
    ) or "(No evidence passed working-memory filters.)"
    system = (
        "Answer the subquery using ONLY the provided evidence excerpts. "
        "If evidence is insufficient, say what is missing. Cite sources by filename."
    )
    user = f"Subquery:\n{subquery}\n\nEvidence:\n{ctx}\n"
    try:
        return normalize_ws(llm.complete_text(system, user, max_tokens=900))
    except Exception:
        return "Unable to generate subquery answer (LLM error)."


def synthesize_final_answer(
    original_query: str,
    episodic_notes: list[str],
    extra_evidence_snippets: list[str],
    llm: LLMProvider | None = None,
) -> str:
    llm = llm or get_llm_provider()
    system = (
        "You synthesize a final research answer from episodic notes and optional evidence. "
        "Structure: short intro, 2-4 paragraphs, brief caveats. "
        "Be explicit when evidence is thin. Do not invent citations."
    )
    notes_block = "\n".join(f"- {n}" for n in episodic_notes)
    ev_block = "\n\n".join(extra_evidence_snippets[:6])
    user = (
        f"Original question:\n{original_query}\n\n"
        f"Episodic summaries:\n{notes_block}\n\n"
        f"Additional evidence excerpts:\n{ev_block}\n"
    )
    try:
        return normalize_ws(llm.complete_text(system, user, max_tokens=1200))
    except Exception:
        return (
            "[Fallback synthesis] "
            + normalize_ws(" ".join(episodic_notes))[:2000]
        )
