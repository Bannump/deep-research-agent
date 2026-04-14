"""
Deterministic, evidence-grounded summarization for demo / mock-LLM mode.

Extracts factual statements from retained chunks only — no hallucinated facts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.utils import normalize_ws

if TYPE_CHECKING:
    from app.retriever import RetrievedChunk

# Prefixes often echoed by LLMs when decomposition JSON is malformed.
_SUBQUERY_PREFIX_PATTERNS = re.compile(
    r"^\s*(?:"
    r"original\s+question|question|subquery|sub-?query|q|user\s+query"
    r")\s*[:.\-–—]\s*",
    re.IGNORECASE,
)

_JUNK_ONLY = re.compile(
    r"^(?:original\s+question|question|subquery)\s*:?\s*$",
    re.IGNORECASE,
)

_MIN_SUBQUERY_LEN = 8
_MAX_EPISODIC_CHARS = 520
_MAX_DIRECT_CHARS = 360


_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of as is are was were be been being
    it its this that these those with from by what which who how when where why
    does do did can could should would will into about over after before then than
    such any all each both few more most other some such only same so than too very
    not no nor also just same than into onto per via
    """.split()
)


def strip_subquery_prefixes(text: str) -> str:
    """Remove leading label fragments like 'Original question:'."""
    s = normalize_ws(text)
    prev = None
    while prev != s:
        prev = s
        s = _SUBQUERY_PREFIX_PATTERNS.sub("", s)
        s = normalize_ws(s)
    return s


def _normalize_for_dedupe(s: str) -> str:
    return re.sub(r"[^\w\s]+", " ", s.lower()).strip()


def sanitize_subqueries(raw: list[str], original_query: str, max_n: int) -> list[str]:
    """
    Drop junk/empty/duplicate subqueries; preserve order.
    If nothing usable remains, caller should fall back to a single cleaned original.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = strip_subquery_prefixes(item)
        if not s or _JUNK_ONLY.match(s):
            continue
        if len(s) < _MIN_SUBQUERY_LEN:
            continue
        if not re.search(r"[a-zA-Z0-9]", s):
            continue
        key = _normalize_for_dedupe(s)
        if len(key) < 6:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    if not out:
        cleaned = strip_subquery_prefixes(original_query)
        if cleaned and len(cleaned) >= _MIN_SUBQUERY_LEN:
            return [cleaned][:max_n]
    return out[:max_n]


def _query_terms(query: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", query.lower())
        if w not in _STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-like units (deterministic)."""
    text = text.replace("\r\n", "\n")
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    for c in chunks:
        s = normalize_ws(c)
        if len(s) > 12:
            out.append(s)
    return out


def _score_sentence(sentence: str, terms: set[str]) -> float:
    low = sentence.lower()
    overlap = sum(1 for t in terms if t in low)
    # Light boost for substantive lines (lists, requirements)
    boost = 0.0
    if re.search(r"\b(must|shall|required|mandatory|AES|TLS|RBAC|HSM|KMS)\b", low):
        boost += 0.5
    return float(overlap) + boost


def grounded_subquery_answer(subquery: str, retained: list[RetrievedChunk]) -> str:
    """Concise answer citing only retained chunk text."""
    if not retained:
        return (
            "No evidence remained in working memory for this subquery after pruning, "
            "so no grounded statements can be made."
        )

    terms = _query_terms(subquery)
    scored: list[tuple[float, int, str, str]] = []  # score, tie-break, sentence, source

    for ti, chunk in enumerate(retained):
        for sent in _split_sentences(chunk.text):
            sc = _score_sentence(sent, terms)
            scored.append((sc, ti, sent, chunk.source))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    seen: set[str] = set()
    bullets: list[tuple[str, str]] = []
    for sc, _ti, sent, src in scored:
        key = normalize_ws(sent.lower())[:220]
        if key in seen:
            continue
        seen.add(key)
        if sc > 0 or len(bullets) < 3:
            bullets.append((sent, src))
        if len(bullets) >= 5:
            break

    if not bullets:
        for chunk in retained:
            for sent in _split_sentences(chunk.text)[:2]:
                bullets.append((sent, chunk.source))
            if bullets:
                break

    lines = [f"- {s} [{src}]" for s, src in bullets[:5]]
    head = (
        f"From the retained excerpts, the evidence relevant to "
        f"'{subquery[:160]}' includes:"
    )
    return normalize_ws(head + "\n\n" + "\n".join(lines))


_BULLET_LINE = re.compile(r"^[-*•]\s+(.*)$")


def grounded_episodic_summary(subquery: str, subquery_answer: str) -> str:
    """
    Shorter than subquery_answer: one tight sentence + optional second clause.
    """
    bullet_texts: list[str] = []
    for line in subquery_answer.splitlines():
        line = line.strip()
        m = _BULLET_LINE.match(line)
        if m:
            inner = re.sub(r"\s*\[[^\]]+\]\s*$", "", m.group(1)).strip()
            if inner:
                bullet_texts.append(inner)
    if bullet_texts:
        core = bullet_texts[0]
        if len(bullet_texts) > 1 and len(core) < 160:
            core = normalize_ws(core + " " + bullet_texts[1])
    else:
        sents = [
            s
            for s in _split_sentences(subquery_answer)
            if "from the retained excerpts" not in s.lower()
        ]
        core = sents[0] if sents else normalize_ws(subquery_answer)[:400]

    core = normalize_ws(core)
    tag = subquery[:50] + ("…" if len(subquery) > 50 else "")
    out = f"[{tag}] {core}"
    if len(out) > _MAX_EPISODIC_CHARS:
        out = out[: _MAX_EPISODIC_CHARS - 3].rstrip() + "..."
    # Episodic tier must stay strictly shorter than the full subquery answer when possible
    if len(subquery_answer) > 40 and len(out) >= len(subquery_answer):
        out = out[: max(20, len(subquery_answer) - 20)].rstrip() + "…"
    return out


def _extract_source_tags(blocks: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    pat = re.compile(r"\[([^\]]+\.(?:md|txt|pdf|html))]", re.IGNORECASE)
    for b in blocks:
        for m in pat.finditer(b):
            s = m.group(1).strip()
            if s and s not in seen:
                seen.add(s)
                found.append(s)
    return found


def grounded_final_answer(
    original_query: str,
    episodic_notes: list[str],
    extra_evidence_snippets: list[str],
) -> str:
    """
    Direct answer first, then compact bullets, then sources — all grounded.
    """
    corpus_parts = list(episodic_notes) + list(extra_evidence_snippets)
    corpus = "\n".join(corpus_parts)
    if not normalize_ws(corpus):
        return (
            "**Direct answer:** The retained evidence for this query is empty or was fully pruned, "
            "so nothing specific can be concluded from the corpus under current memory limits."
        )

    terms = _query_terms(original_query)
    sentences: list[str] = []
    for block in corpus_parts:
        sentences.extend(_split_sentences(block))
    if not sentences:
        sentences = [normalize_ws(corpus[:500])]

    ranked = [(_score_sentence(s, terms), s) for s in sentences]
    ranked.sort(key=lambda x: (-x[0], x[1]))

    # Direct answer: best overlapping factual sentences, 1–2 sentences
    direct_parts: list[str] = []
    for sc, sent in ranked:
        if sc > 0 and sent not in direct_parts:
            direct_parts.append(sent)
        if len(normalize_ws(" ".join(direct_parts))) >= _MAX_DIRECT_CHARS:
            break
    if not direct_parts:
        for _sc, sent in ranked[:2]:
            if sent not in direct_parts:
                direct_parts.append(sent)
            if len(direct_parts) >= 2:
                break

    direct = normalize_ws(" ".join(direct_parts[:2]))
    if not direct:
        direct = normalize_ws(sentences[0][: _MAX_DIRECT_CHARS])
    if len(direct) > _MAX_DIRECT_CHARS:
        direct = direct[: _MAX_DIRECT_CHARS - 1].rstrip() + "…"

    # Bullets: next distinct high-signal sentences (avoid duplicating direct)
    bullet_sents: list[str] = []
    seen_low = {normalize_ws(direct.lower())[:120]}
    for sc, sent in ranked:
        sl = normalize_ws(sent.lower())[:120]
        if sl in seen_low:
            continue
        seen_low.add(sl)
        if sent != direct and sent not in bullet_sents:
            bullet_sents.append(sent)
        if len(bullet_sents) >= 4:
            break

    if not bullet_sents:
        for _sc, sent in ranked[2:6]:
            sl = normalize_ws(sent.lower())[:120]
            if sl not in seen_low:
                bullet_sents.append(sent)
                seen_low.add(sl)

    bullets_md = "\n".join(f"- {normalize_ws(s)}" for s in bullet_sents[:5])
    sources = _extract_source_tags(corpus_parts)
    if not sources:
        # Fall back to filenames appearing in bracket citations from subquery answers
        for note in episodic_notes:
            sources.extend(_extract_source_tags([note]))
        sources = list(dict.fromkeys(sources))

    src_line = ""
    if sources:
        src_line = "\n\n**Sources (from evidence tags):** " + "; ".join(sources[:12])

    caveat = ""
    if not any(_score_sentence(s, terms) > 0 for s in sentences):
        caveat = (
            "\n\n*(Qualified: overlap between your phrasing and the retained excerpts is weak; "
            "the bullets above are still drawn only from the evidence shown.)*"
        )

    return normalize_ws(
        f"**Direct answer:** {direct}\n\n"
        f"**Supporting details (from retained evidence):**\n{bullets_md}"
        f"{caveat}{src_line}"
    )
