"""Final synthesis from episodic summaries + selected evidence."""

from __future__ import annotations

from app.evidence_clean import clean_chunk_for_synthesis
from app.grounded_synthesis import (
    grounded_final_answer,
    grounded_subquery_answer,
    strip_user_facing_subquery_labels,
)
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

    cleaned = [clean_chunk_for_synthesis(c.text) or c.text for c in retained]
    ctx = "\n\n".join(
        f"[{c.source}] (chunk {c.chunk_id})\n{t}" for c, t in zip(retained, cleaned)
    ) or "(No evidence passed working-memory filters.)"
    system = (
        "Answer the subquery using ONLY the provided evidence excerpts. "
        "Write 2–5 compact sentences (or a short grouped bullet list) with concrete controls, "
        "requirements, or findings. Do not copy markdown headings as facts. "
        "Stay aligned with this subquery's topic; do not paste unrelated categories. "
        "Never output bracketed subquery labels like [What ...?]. "
        "If evidence is insufficient, say what is missing. Cite sources by filename. "
        "Do not claim anything not supported by the excerpts."
    )
    user = f"Subquery:\n{subquery}\n\nEvidence:\n{ctx}\n"
    try:
        return strip_user_facing_subquery_labels(
            normalize_ws(llm.complete_text(system, user, max_tokens=900))
        )
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
        "Begin with 1-2 sentences that summarize the major themes (e.g., technical, audit/logging, governance, risks) "
        "— not a single narrow detail. "
        "For control-focused questions, prioritize concrete controls first, then risks/gaps in a separate section. "
        "Do not treat a risk or gap as if it were a control. "
        "Deduplicate: do not repeat the same requirement in different wording. "
        "Never output bracketed subquery labels like [What ...?] or stitched internal scaffolding. "
        "Use plain text section labels such as 'Key controls' and 'Risks and enforcement gaps' (no markdown # headings). "
        "If evidence is partial, say so explicitly. Do not invent facts or filenames."
    )
    notes_block = "\n".join(f"- {n}" for n in episodic_notes)
    cleaned_snips = []
    for s in extra_evidence_snippets[:12]:
        c = clean_chunk_for_synthesis(s) if "#" in s else s
        cleaned_snips.append(c or s)
    ev_block = "\n\n".join(cleaned_snips)
    user = (
        f"Original question:\n{original_query}\n\n"
        f"Episodic summaries:\n{notes_block}\n\n"
        f"Additional evidence excerpts:\n{ev_block}\n"
    )
    try:
        return strip_user_facing_subquery_labels(
            normalize_ws(llm.complete_text(system, user, max_tokens=1200))
        )
    except Exception:
        return grounded_final_answer(original_query, episodic_notes, extra_evidence_snippets)
