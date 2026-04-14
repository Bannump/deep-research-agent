"""Final synthesis from episodic summaries + selected evidence."""

from __future__ import annotations

from app.grounded_synthesis import grounded_final_answer, grounded_subquery_answer
from app.llm import LLMProvider, MockLLMProvider, get_llm_provider
from app.retriever import RetrievedChunk
from app.utils import normalize_ws


def _use_grounded(llm: LLMProvider | None) -> bool:
    """Offline / demo path: extractive synthesis only (no API keys required)."""
    prov = llm or get_llm_provider()
    return isinstance(prov, MockLLMProvider)


def answer_subquery(
    subquery: str,
    retained: list[RetrievedChunk],
    llm: LLMProvider | None = None,
) -> str:
    """Generate an answer for one subquery using only retained evidence."""
    llm = llm or get_llm_provider()
    if _use_grounded(llm):
        return grounded_subquery_answer(subquery, retained)

    ctx = "\n\n".join(
        f"[{c.source}] (chunk {c.chunk_id})\n{c.text}" for c in retained
    ) or "(No evidence passed working-memory filters.)"
    system = (
        "Answer the subquery using ONLY the provided evidence excerpts. "
        "Use concrete controls, requirements, or findings from the text. "
        "If evidence is insufficient, say what is missing. Cite sources by filename. "
        "Do not claim anything not supported by the excerpts."
    )
    user = f"Subquery:\n{subquery}\n\nEvidence:\n{ctx}\n"
    try:
        return normalize_ws(llm.complete_text(system, user, max_tokens=900))
    except Exception:
        return grounded_subquery_answer(subquery, retained)


def synthesize_final_answer(
    original_query: str,
    episodic_notes: list[str],
    extra_evidence_snippets: list[str],
    llm: LLMProvider | None = None,
) -> str:
    llm = llm or get_llm_provider()
    if _use_grounded(llm):
        return grounded_final_answer(original_query, episodic_notes, extra_evidence_snippets)

    system = (
        "You synthesize a final research answer from episodic notes and evidence excerpts. "
        "Start with a direct answer in 1-2 sentences using concrete facts (controls, tools, requirements). "
        "Then add a short section with bullet points for supporting details. "
        "Mention specific technologies or processes when they appear in the text. "
        "If evidence is partial, qualify your statements. Do not invent facts or filenames."
    )
    notes_block = "\n".join(f"- {n}" for n in episodic_notes)
    ev_block = "\n\n".join(extra_evidence_snippets[:12])
    user = (
        f"Original question:\n{original_query}\n\n"
        f"Episodic summaries:\n{notes_block}\n\n"
        f"Additional evidence excerpts:\n{ev_block}\n"
    )
    try:
        return normalize_ws(llm.complete_text(system, user, max_tokens=1200))
    except Exception:
        return grounded_final_answer(original_query, episodic_notes, extra_evidence_snippets)
