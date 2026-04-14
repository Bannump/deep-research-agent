"""Normalize and filter evidence text before summarization (no headings-as-facts)."""

from __future__ import annotations

import re

from app.utils import normalize_ws

# Lines that are only markdown headings / title noise
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BOLD_WRAPPER = re.compile(r"^\s*\*\*[^*]+\*\*\s*:?\s*$")
_ONLY_LIST_MARKER = re.compile(r"^\s*[-*•]\s*$")
_PUNCT_ONLY = re.compile(r"^[\s\-–—:;,.!?*#`|\\/\[\](){}'\"]+$")


def collapse_repeated_punctuation(text: str) -> str:
    """Collapse repeated ? ! and trim duplicate boundary punctuation."""
    s = text.strip()
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r"([?!])\1+", r"\1", s)
    return normalize_ws(s)


def is_markdown_heading_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if _MD_HEADING.match(t):
        return True
    if re.match(r"^#{1,6}\s*\S", t) and len(t) < 120 and not re.search(r"[.!?]\s", t):
        return True
    return False


def is_section_title_only(line: str) -> bool:
    """Heuristic: short title case line without sentence structure."""
    t = normalize_ws(line)
    if len(t) < 4 or len(t) > 100:
        return False
    if ":" in t and t.count(" ") < 2:
        return True
    if t.isupper() and " " in t:
        return True
    return False


def clean_evidence_line(line: str) -> str | None:
    """
    Return a cleaned factual line, or None if the line should be dropped.
    Removes heading markers from inline text where possible.
    """
    raw = line.replace("\r\n", "\n").strip()
    if not raw:
        return None
    if _ONLY_LIST_MARKER.match(raw):
        return None
    if _PUNCT_ONLY.match(raw):
        return None

    t = raw
    # Strip leading markdown heading markers but keep remainder if substantive
    t = _MD_HEADING.sub("", t).strip()
    t = re.sub(r"^\*\*([^*]+)\*\*\s*", r"\1 ", t).strip()

    if not t or _PUNCT_ONLY.match(t):
        return None
    if is_markdown_heading_line(raw) and len(t) < 30 and " " not in t:
        return None
    # Heading stubs like "## Risks" normalize to short titles ("Risks") without sentence punctuation.
    if (
        raw.lstrip().startswith("#")
        and is_markdown_heading_line("# " + t)
        and len(t.split()) <= 4
        and not re.search(r"[.!?]", t)
    ):
        return None
    if _BOLD_WRAPPER.match(raw):
        return None

    low = t.lower()
    if low in ("technical controls", "risks and recommendations", "recommendations", "overview"):
        return None

    if len(t) < 12:
        # Allow short concrete tokens like "AES-256" only if alphanumeric heavy
        if not re.search(r"[A-Z0-9]{3}", t):
            return None

    if not re.search(r"[a-zA-Z0-9]", t):
        return None

    return normalize_ws(t)


def clean_chunk_for_synthesis(text: str) -> str:
    """Remove heading-only lines and markdown debris; keep paragraph text."""
    if not text.strip():
        return ""
    lines_out: list[str] = []
    for line in text.splitlines():
        cl = clean_evidence_line(line)
        if cl:
            lines_out.append(cl)
    if not lines_out:
        # Fallback: strip # headers in bulk
        flat = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
        flat = normalize_ws(flat)
        return flat if len(flat) > 20 else ""
    return "\n".join(lines_out)


def dedupe_near_identical_lines(lines: list[str], max_items: int) -> list[str]:
    """Order-preserving dedupe using normalized keys."""
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = re.sub(r"[^\w\s]+", " ", line.lower()).strip()
        key = re.sub(r"\s+", " ", key)[:200]
        if len(key) < 8:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_items:
            break
    return out


_STOPWORDS_DEDUPE = frozenset(
    """
    a an the and or but if in on at to for of as is are was were be been being
    it its this that these those with from by what which who how when where why
    does do did can could should would will into about over after before then than
    such any all each both few more most other some only same so than too very
    not no nor also just per via from
    """.split()
)


def _word_bag(s: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", s.lower())
        if w not in _STOPWORDS_DEDUPE
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedupe_sentences_semantic(
    sentences: list[str],
    max_items: int,
    *,
    overlap_threshold: float = 0.52,
) -> list[str]:
    """
    Order-preserving dedupe for near-duplicate facts (reordered wording, repeated claims).
    Uses Jaccard overlap on word bags; keeps the first occurrence.
    """
    out: list[str] = []
    bags: list[set[str]] = []
    for sent in sentences:
        ws = _word_bag(sent)
        if len(ws) < 2:
            key = re.sub(r"[^\w\s]+", " ", sent.lower()).strip()
            key = re.sub(r"\s+", " ", key)[:200]
            if len(key) < 8:
                continue
            dup = any(key == re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", o.lower()).strip())[:200] for o in out)
            if dup:
                continue
            out.append(sent)
            bags.append(ws)
            if len(out) >= max_items:
                break
            continue
        if any(_jaccard(ws, prev) >= overlap_threshold for prev in bags):
            continue
        out.append(sent)
        bags.append(ws)
        if len(out) >= max_items:
            break
    return out
