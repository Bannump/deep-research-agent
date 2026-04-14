"""
Deterministic, evidence-grounded summarization for demo / mock-LLM mode.

Extracts factual statements from retained chunks only — no hallucinated facts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from app.evidence_clean import (
    clean_chunk_for_synthesis,
    clean_evidence_line,
    collapse_repeated_punctuation,
    dedupe_near_identical_lines,
    dedupe_sentences_semantic,
)
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
_MAX_EPISODIC_CHARS = 420
_MAX_DIRECT_CHARS = 420

# Bloated wrapper patterns the planner must not emit (substring match after lowercasing).
_BANNED_SUBQUERY_PATTERNS = (
    "what definitions and background apply to",
    "what mechanisms, processes, or constraints are described regarding",
    "what outcomes, risks, or recommendations are stated about",
)

_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of as is are was were be been being
    it its this that these those with from by what which who how when where why
    does do did can could should would will into about over after before then than
    such any all each both few more most other some such only same so than too very
    not no nor also just same than into onto per via
    """.split()
)

SentenceKind = Literal["control", "risk", "neutral"]
SubqueryTopic = Literal["technical", "governance", "risk", "neutral"]

# Subquery labels from the planner look like "[What technical security controls are described?] : ..."
# Do not strip source citations like "[02_controls.md]" (no "?" inside the brackets).
_SUBQUERY_Q_BRACKET = re.compile(r"\[[^\]]*\?\]\s*[:.\-–—]?\s*")


def strip_user_facing_subquery_labels(text: str) -> str:
    """
    Remove bracketed subquery scaffolding from user-facing prose.
    Targets question-shaped labels only so evidence filenames in brackets are preserved.
    """
    s = text
    prev = None
    while prev != s:
        prev = s
        s = _SUBQUERY_Q_BRACKET.sub("", s)
        s = normalize_ws(s)
    return s


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


def _sanitize_single_subquery_item(item: str) -> str | None:
    if not isinstance(item, str):
        return None
    s = strip_subquery_prefixes(item)
    s = collapse_repeated_punctuation(s)
    if not s or _JUNK_ONLY.match(s):
        return None
    low = s.lower()
    if any(p in low for p in _BANNED_SUBQUERY_PATTERNS):
        return None
    if len(s) < _MIN_SUBQUERY_LEN:
        return None
    if not re.search(r"[a-zA-Z0-9]", s):
        return None
    # Reject overly generic single-token questions
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", low) if w not in _STOPWORDS]
    if len(words) < 2 and "?" not in s:
        return None
    return s


def sanitize_subqueries(raw: list[str], original_query: str, max_n: int) -> list[str]:
    """
    Drop junk/empty/duplicate/banned-pattern subqueries; preserve order.
    If nothing usable remains, caller should fall back to a single cleaned original.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        s = _sanitize_single_subquery_item(item)
        if not s:
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
        cleaned = collapse_repeated_punctuation(strip_subquery_prefixes(original_query))
        if cleaned and len(cleaned) >= _MIN_SUBQUERY_LEN:
            return [cleaned][:max_n]
    return out[:max_n]


def _core_topic_phrase(query: str) -> str:
    """Strip trailing boilerplate like 'in the sample corpus' for fallback scaffolding."""
    s = strip_subquery_prefixes(query)
    s = collapse_repeated_punctuation(s).rstrip("?").strip()
    s = re.sub(
        r"\s+in\s+the\s+sample\s+corpus\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180] if s else ""


def intent_aware_fallback_subqueries(cleaned_query: str, max_n: int) -> list[str]:
    """
    Deterministic, retrieval-friendly subqueries without bloated wrapper templates.
    """
    base = _core_topic_phrase(cleaned_query) or cleaned_query.strip()
    low = base.lower()

    # Control / security posture questions
    if any(
        k in low
        for k in (
            "control",
            "controls",
            "safeguard",
            "security posture",
            "compliance",
            "encryption",
            "key management",
        )
    ):
        out = [
            "What technical security controls are described?",
            "What governance or review controls are described?",
            "What risks, enforcement gaps, or recommendations are mentioned?",
        ]
        return out[:max_n]

    if any(k in low for k in ("risk", "risks", "failure", "gap", "recommendation", "remediation")):
        out = [
            "What risks or failure modes are described?",
            "What recommendations or remediations are stated?",
            "What related controls or assurance practices are mentioned?",
        ]
        return out[:max_n]

    # Generic decomposition: still avoid banned wrappers
    tail = base if len(base) <= 140 else base[:137].rstrip() + "…"
    out = [
        f"What concrete requirements or controls apply to {tail}?".replace("??", "?"),
        f"What operational or governance practices are described for {tail}?".replace("??", "?"),
        f"What risks, gaps, or follow-up actions are mentioned for {tail}?".replace("??", "?"),
    ]
    return out[:max_n]


def is_cleaned_original_passthrough(subqueries: list[str], cleaned_original: str) -> bool:
    """True when decomposition collapsed to a single subquery equal to the cleaned original."""
    if len(subqueries) != 1:
        return False
    return _normalize_for_dedupe(subqueries[0]) == _normalize_for_dedupe(cleaned_original)


def subqueries_quality_poor(subqueries: list[str], original_query: str) -> bool:
    """True if decomposition is empty, duplicate-heavy, or only banned wrappers."""
    if not subqueries:
        return True
    okey = _normalize_for_dedupe(original_query)
    keys = [_normalize_for_dedupe(s) for s in subqueries]
    if len(set(keys)) < len(keys):
        return True
    if all(len(k) < 10 for k in keys):
        return True
    banned_hits = sum(
        1 for s in subqueries if any(p in s.lower() for p in _BANNED_SUBQUERY_PATTERNS)
    )
    if banned_hits == len(subqueries):
        return True
    # Near-duplicate of the full question repeated for each slot
    if len(subqueries) >= 2 and all(_normalize_for_dedupe(s) == okey for s in subqueries):
        return True
    return False


def _query_terms(query: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", query.lower())
        if w not in _STOPWORDS
    }


def subquery_topic_bucket(subquery: str) -> SubqueryTopic:
    """Coarse intent for retrieval alignment (technical vs governance vs risk)."""
    low = subquery.lower()
    if re.search(
        r"\b(risks?|enforcement gaps?|recommendations?|failure modes?|remediation)\b",
        low,
    ) and "technical security" not in low:
        return "risk"
    if "technical security" in low or (
        "encryption" in low and "governance" not in low and "review" not in low
    ):
        return "technical"
    if "governance" in low or "review controls" in low or (
        "review" in low and "technical" not in low and "risk" not in low
    ):
        return "governance"
    return "neutral"


def _has_crypto_transport(sentence: str) -> bool:
    low = sentence.lower()
    return bool(
        re.search(r"\b(aes|tls|ssl|encrypt|cipher|kms|hsm|key rotation|in transit|at rest)\b", low)
    )


def _has_logging_audit(sentence: str) -> bool:
    low = sentence.lower()
    return "log" in low and (
        "central" in low or "audit" in low or "mandatory" in low or "siem" in low or "immutable" in low
    )


def _has_governance_anchor(sentence: str) -> bool:
    low = sentence.lower()
    return any(
        h in low
        for h in (
            "governance",
            "review",
            "approval",
            "vendor",
            "minimiz",
            "committee",
            "policy review",
            "break-glass",
            "break glass",
            "emergency access",
            "attestation",
            "quarterly",
            "annual",
        )
    )


def _has_strong_risk_signal(sentence: str) -> bool:
    low = sentence.lower()
    return any(
        h in low
        for h in (
            "stale polic",
            "shadow pipeline",
            "alert fatigue",
            "failure mode",
            "enforcement gap",
            "not enforced",
            "bypass official",
            "undeployed",
        )
    )


def _topic_alignment_adjustment(sentence: str, bucket: SubqueryTopic) -> float:
    """Nudge scoring so subquery answers stay on-topic (reduce category leakage)."""
    if bucket == "neutral":
        return 0.0
    adj = 0.0
    if bucket == "technical":
        if _has_crypto_transport(sentence) or _has_logging_audit(sentence):
            adj += 1.6
        if re.search(r"\b(rbac|abac|mfa|sso|privileged|least privilege)\b", sentence.lower()):
            adj += 1.1
        if _has_governance_anchor(sentence) and not (
            _has_crypto_transport(sentence) or _has_logging_audit(sentence)
        ):
            adj -= 1.1
        if _has_strong_risk_signal(sentence) and not re.search(
            r"\b(must|shall|required|mandatory|control)\b", sentence.lower()
        ):
            adj -= 1.4
    elif bucket == "governance":
        if _has_governance_anchor(sentence):
            adj += 1.8
        if _has_crypto_transport(sentence) and not _has_governance_anchor(sentence):
            adj -= 1.7
        if _has_logging_audit(sentence) and not _has_governance_anchor(sentence):
            adj -= 0.9
        if _has_strong_risk_signal(sentence):
            adj -= 1.2
    elif bucket == "risk":
        if _has_strong_risk_signal(sentence):
            adj += 1.9
        if re.search(r"\b(recommend|remediat|mitigat|should\b|need to\b)\b", sentence.lower()):
            adj += 0.6
        if _has_crypto_transport(sentence) and not _has_strong_risk_signal(sentence):
            adj -= 1.0
    return adj


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-like units (deterministic)."""
    text = text.replace("\r\n", "\n")
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    for c in chunks:
        s = normalize_ws(c)
        cl = clean_evidence_line(s) if s else None
        if not cl:
            continue
        # List / extraction debris like ": Centralized logging..."
        cl = re.sub(r"^\s*:\s+", "", cl).strip()
        if len(cl) > 12:
            out.append(cl)
    return out


def _classify_sentence_kind(sentence: str) -> SentenceKind:
    low = sentence.lower()
    strong_risk = (
        "stale polic",
        "shadow pipeline",
        "alert fatigue",
        "failure mode",
        "undeployed",
        "not enforced",
        "bypass official",
        "enforcement gap",
        "weak overlap",
    )
    if any(h in low for h in strong_risk):
        return "risk"
    soft_risk = (
        " risk ",
        "risks ",
        "failure modes",
        " gap ",
        "gaps ",
        "however, ",
        "caveat",
    )
    if any(h in low for h in soft_risk) and not re.search(
        r"\b(must|shall|required|mandatory)\b", low
    ):
        return "risk"

    literal_control = (
        "required",
        "mandatory",
        "encrypt",
        "aes-",
        "aes ",
        "tls ",
        "rbac",
        "abac",
        "hsm",
        "kms",
        "rotation",
        "logging",
        "audit-grade",
        "audit ",
        "checklist",
        "policy-as-code",
        "conformance test",
        "sampled audit",
        "least privilege",
        "break-glass",
    )
    if any(h in low for h in literal_control):
        return "control"
    if re.search(r"\b(must|shall|required|mandatory)\b", low):
        return "control"
    if re.search(r"\bcontrols?\b", low) and "not enforced" not in low and "documented but" not in low:
        return "control"
    return "neutral"


def query_is_control_focused(original_query: str) -> bool:
    low = original_query.lower()
    return any(
        k in low
        for k in (
            "control",
            "controls",
            "safeguard",
            "requirement",
            "encryption",
            "governance",
            "assurance",
        )
    )


def query_is_risk_focused(original_query: str) -> bool:
    """Prefer risks/gaps in synthesis when the question is not primarily about controls."""
    if query_is_control_focused(original_query):
        return False
    low = original_query.lower()
    return any(
        k in low
        for k in (
            "risk",
            "risks",
            "gap",
            "gaps",
            "failure",
            "recommendation",
            "remediation",
            "enforcement",
        )
    )


def _control_corpus_theme_clauses(controls: list[str], neutral: list[str]) -> list[str]:
    """Short theme clauses for a multi-category opening (control-focused questions)."""
    pool = controls + neutral
    clauses: list[str] = []
    if any(_has_crypto_transport(s) or "encrypt" in s.lower() for s in pool[:40]):
        clauses.append("technical protections (encryption, transport, and key-handling expectations)")
    if any(_has_logging_audit(s) for s in pool[:40]):
        clauses.append("auditability and centralized logging expectations")
    if any(_has_governance_anchor(s) for s in pool[:40]):
        clauses.append("governance- and review-oriented safeguards")
    if any("operational" in s.lower() or "on-call" in s.lower() or "runbook" in s.lower() for s in pool[:40]):
        clauses.append("operational safeguards")
    # De-dupe clause text
    seen: set[str] = set()
    out: list[str] = []
    for c in clauses:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _synthesize_control_focused_opener(
    controls: list[str],
    neutral: list[str],
    risks: list[str],
    terms: set[str],
    prefer: SentenceKind | None,
) -> str:
    """1–2 sentences: theme-level synthesis, not a single narrow fact."""
    themes = _control_corpus_theme_clauses(controls, neutral)
    has_risk_signal = any(_has_strong_risk_signal(s) for s in risks[:30])
    if len(themes) >= 2:
        lead = (
            "Across the retained excerpts, the sample corpus emphasizes "
            + ", ".join(themes[:-1])
            + ", and "
            + themes[-1]
            + "."
        )
    elif len(themes) == 1:
        lead = f"Across the retained excerpts, the strongest theme is {themes[0]}."
    else:
        lead = ""

    if not lead:
        pool = dedupe_sentences_semantic(controls + neutral, 6)
        ranked = sorted(
            ((_score_sentence(s, terms, prefer), s) for s in pool),
            key=lambda x: (-x[0], x[1]),
        )
        parts = [s for _, s in ranked[:2]]
        parts = dedupe_sentences_semantic(parts, 2)
        lead = normalize_ws(" ".join(parts)) if parts else ""

    if has_risk_signal and lead:
        lead += " Related passages also call out enforcement gaps and residual risks, summarized separately below."
    elif has_risk_signal:
        lead = (
            "The retained excerpts highlight enforcement gaps and residual risks, "
            "with supporting control detail captured below."
        )

    lead = normalize_ws(lead)
    if len(lead) > _MAX_DIRECT_CHARS:
        lead = lead[: _MAX_DIRECT_CHARS - 1].rstrip() + "…"
    return lead


def _synthesize_risk_focused_opener(risks: list[str], terms: set[str], prefer: SentenceKind | None) -> str:
    pool = dedupe_sentences_semantic(risks, 8)
    ranked = sorted(
        ((_score_sentence(s, terms, prefer), s) for s in pool),
        key=lambda x: (-x[0], x[1]),
    )
    parts: list[str] = []
    for _sc, s in ranked:
        parts.append(s)
        if len(parts) >= 2:
            break
    out = normalize_ws(" ".join(parts))
    if len(out) > _MAX_DIRECT_CHARS:
        out = out[: _MAX_DIRECT_CHARS - 1].rstrip() + "…"
    return out


def _score_sentence(
    sentence: str,
    terms: set[str],
    prefer: SentenceKind | None,
    *,
    topic_bucket: SubqueryTopic | None = None,
) -> float:
    low = sentence.lower()
    overlap = sum(1 for t in terms if t in low)
    boost = 0.0
    if re.search(r"\b(must|shall|required|mandatory)\b", low):
        boost += 1.2
    if re.search(r"\b(AES|TLS|RBAC|HSM|KMS|ABAC)\b", low):
        boost += 0.8
    kind = _classify_sentence_kind(sentence)
    if prefer == "control":
        if kind == "control":
            boost += 1.5
        elif kind == "risk":
            boost -= 1.2
        elif kind == "neutral":
            boost += 0.2
    elif prefer == "risk":
        if kind == "risk":
            boost += 1.4
        elif kind == "control":
            boost -= 1.0
    else:
        if kind == "control":
            boost += 0.6
        elif kind == "risk":
            boost += 0.2
    base = float(overlap) + boost
    if topic_bucket and topic_bucket != "neutral":
        base += _topic_alignment_adjustment(sentence, topic_bucket)
    return base


def grounded_subquery_answer(subquery: str, retained: list[RetrievedChunk]) -> str:
    """Compact local answer: 2–5 sentences grounded in retained text only."""
    if not retained:
        return (
            "No evidence remained in working memory for this subquery after pruning, "
            "so no grounded statements can be made."
        )

    cleaned_blocks: list[str] = []
    for chunk in retained:
        ctext = clean_chunk_for_synthesis(chunk.text)
        if ctext:
            cleaned_blocks.append(ctext)

    if not cleaned_blocks:
        return (
            "Retained chunks contained only headings or empty fragments after cleaning, "
            "so no grounded statements can be made."
        )

    terms = _query_terms(subquery)
    low_sq = subquery.lower()
    if re.search(r"\b(risks?|enforcement|gaps?|recommendations?|failure)\b", low_sq):
        prefer: SentenceKind | None = "risk"
    else:
        prefer = "control"
    bucket = subquery_topic_bucket(subquery)

    scored: list[tuple[float, int, str]] = []
    for ti, block in enumerate(cleaned_blocks):
        for sent in _split_sentences(block):
            sc = _score_sentence(sent, terms, prefer, topic_bucket=bucket)
            scored.append((sc, ti, sent))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    picked: list[str] = []
    seen: set[str] = set()
    for sc, _ti, sent in scored:
        if sc < -0.5:
            continue
        key = normalize_ws(sent.lower())[:220]
        if key in seen:
            continue
        seen.add(key)
        picked.append(sent)
        if len(picked) >= 5:
            break

    if len(picked) < 2:
        for block in cleaned_blocks:
            for sent in _split_sentences(block):
                if sent not in picked:
                    picked.append(sent)
                if len(picked) >= 3:
                    break
            if len(picked) >= 3:
                break

    picked = dedupe_near_identical_lines(picked, 5)
    body = normalize_ws(" ".join(picked[:5]))
    if len(body) > 900:
        body = body[:897].rstrip() + "…"

    lead = (
        f"For this subquery, the retained evidence supports the following "
        f"(sources: {', '.join(sorted({c.source for c in retained}))}):"
    )
    return normalize_ws(lead + " " + body)


def _episodic_fingerprint(text: str) -> str:
    return re.sub(r"[^\w\s]+", " ", text.lower()).strip()[:160]


def grounded_episodic_summary(
    subquery: str,
    subquery_answer: str,
    prior_episodic: list[str] | None = None,
) -> str:
    """
    One or two lines, strictly shorter than subquery_answer, distinct per subquery angle.
    """
    prior_episodic = prior_episodic or []
    # Strip boilerplate lead from answer for compression
    core_text = subquery_answer
    low_ans = subquery_answer.lower()
    if "retained evidence supports" in low_ans:
        idx = low_ans.find("following")
        if idx != -1:
            core_text = subquery_answer[idx + len("following") :].strip(" :")
    core_text = re.sub(r"^\(sources:[^)]+\)\s*", "", core_text, flags=re.IGNORECASE)

    sents = [s for s in _split_sentences(core_text) if len(s) > 20]
    if not sents:
        sents = [normalize_ws(core_text)[:400]]

    terms = _query_terms(subquery)
    low_sq = subquery.lower()
    if re.search(r"\b(risks?|enforcement|gaps?|recommendations?|failure)\b", low_sq):
        prefer_ep: SentenceKind | None = "risk"
    else:
        prefer_ep = "control"
    bucket = subquery_topic_bucket(subquery)
    ranked = [(_score_sentence(s, terms, prefer_ep, topic_bucket=bucket), s) for s in sents]
    ranked.sort(key=lambda x: (-x[0], x[1]))

    prior_fps = {_episodic_fingerprint(p) for p in prior_episodic}

    chosen = ""
    for _sc, sent in ranked:
        fp = _episodic_fingerprint(sent)
        if fp in prior_fps:
            continue
        chosen = sent
        break
    if not chosen and ranked:
        chosen = ranked[0][1]

    out = normalize_ws(strip_user_facing_subquery_labels(chosen))
    if not out:
        out = normalize_ws(strip_user_facing_subquery_labels(ranked[0][1])) if ranked else ""
    if len(out) > _MAX_EPISODIC_CHARS:
        out = out[: _MAX_EPISODIC_CHARS - 3].rstrip() + "…"

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
    Theme-level synthesis first, then compact grouped bullets — grounded, plain text section labels.
    """
    corpus_parts: list[str] = []
    for b in list(episodic_notes) + list(extra_evidence_snippets):
        raw = strip_user_facing_subquery_labels(b or "")
        c = clean_chunk_for_synthesis(raw)
        if not c:
            c = normalize_ws(raw)
        if c:
            corpus_parts.append(c)

    corpus = "\n".join(corpus_parts)
    if not normalize_ws(corpus):
        return (
            "The retained evidence for this query is empty or was fully pruned, "
            "so nothing specific can be concluded from the corpus under current memory limits."
        )

    control_focused = query_is_control_focused(original_query)
    risk_focused = query_is_risk_focused(original_query)
    terms = _query_terms(original_query)
    prefer: SentenceKind | None
    if control_focused:
        prefer = "control"
    elif risk_focused:
        prefer = "risk"
    else:
        prefer = None

    sentences: list[str] = []
    for block in corpus_parts:
        sentences.extend(_split_sentences(block))
    sentences = [strip_user_facing_subquery_labels(s) for s in sentences]
    sentences = [s for s in sentences if s]
    if not sentences:
        sentences = [normalize_ws(corpus[:500])]

    ranked = [(_score_sentence(s, terms, prefer), s) for s in sentences]
    ranked.sort(key=lambda x: (-x[0], x[1]))

    controls: list[str] = []
    risks: list[str] = []
    neutral: list[str] = []
    for _sc, sent in ranked:
        k = _classify_sentence_kind(sent)
        if k == "control":
            controls.append(sent)
        elif k == "risk":
            risks.append(sent)
        else:
            neutral.append(sent)

    if control_focused:
        direct = _synthesize_control_focused_opener(controls, neutral, risks, terms, prefer)
        if not normalize_ws(direct) and ranked:
            direct = normalize_ws(ranked[0][1])
            if len(direct) > _MAX_DIRECT_CHARS:
                direct = direct[: _MAX_DIRECT_CHARS - 1].rstrip() + "…"
    elif risk_focused:
        direct = _synthesize_risk_focused_opener(risks, terms, prefer)
        if not normalize_ws(direct) and ranked:
            direct = normalize_ws(ranked[0][1])
    else:
        direct_pool = [s for _, s in ranked]
        direct_parts: list[str] = []
        seen_d: set[str] = set()
        for sent in direct_pool:
            sl = normalize_ws(sent.lower())[:120]
            if sl in seen_d:
                continue
            seen_d.add(sl)
            direct_parts.append(sent)
            if len(normalize_ws(" ".join(direct_parts))) >= _MAX_DIRECT_CHARS // 2:
                if len(direct_parts) >= 2:
                    break
            if len(direct_parts) >= 2:
                break
        if not direct_parts:
            direct_parts = [ranked[0][1]]
        direct = normalize_ws(" ".join(direct_parts[:2]))
        if len(direct) > _MAX_DIRECT_CHARS:
            direct = direct[: _MAX_DIRECT_CHARS - 1].rstrip() + "…"

    direct_norm: set[str] = set()
    for s in _split_sentences(direct):
        direct_norm.add(normalize_ws(s.lower())[:160])

    def _norm_key(s: str) -> str:
        return normalize_ws(s.lower())[:140]

    def _pick_bullets(pool: list[str], cap: int, exclude: set[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for sent in pool:
            sl = _norm_key(sent)
            if sl in seen or sl in exclude:
                continue
            seen.add(sl)
            out.append(sent)
            if len(out) >= cap * 2:
                break
        deduped = dedupe_sentences_semantic(out, cap)
        return deduped

    opener_keys = {_norm_key(sent) for sent in _split_sentences(direct)}

    exclude_for_ctrl = set(direct_norm) | opener_keys

    ctrl_candidates = dedupe_sentences_semantic(controls + neutral, 24)
    risk_candidates = dedupe_sentences_semantic(risks, 20)

    ctrl_lines = _pick_bullets(ctrl_candidates, 6, exclude_for_ctrl)
    exclude_after_ctrl = exclude_for_ctrl | {_norm_key(x) for x in ctrl_lines}
    risk_lines = _pick_bullets(risk_candidates, 5, exclude_after_ctrl)

    extra_pool = [s for s in neutral if s not in ctrl_lines]
    extra_neutral = _pick_bullets(extra_pool, 3, exclude_for_ctrl | {_norm_key(x) for x in ctrl_lines})

    weak = not any(_score_sentence(s, terms, prefer) > 0 for s in sentences)
    caveat = ""
    if weak:
        caveat = (
            " Note: overlap between your phrasing and the retained excerpts is weak; "
            "the statements below are still drawn only from the evidence shown."
        )

    lines: list[str] = [f"{direct}{caveat}"]

    if ctrl_lines:
        lines.append("")
        lines.append("Key controls (from retained evidence):")
        for s in dedupe_near_identical_lines(ctrl_lines, 8):
            lines.append(f"- {normalize_ws(s)}")

    if risk_lines:
        lines.append("")
        lines.append("Risks and enforcement gaps (from retained evidence):")
        for s in dedupe_near_identical_lines(risk_lines, 6):
            lines.append(f"- {normalize_ws(s)}")

    if extra_neutral and (not ctrl_lines or len(ctrl_lines) < 2):
        lines.append("")
        lines.append("Additional context (from retained evidence):")
        for s in dedupe_near_identical_lines(extra_neutral, 3):
            lines.append(f"- {normalize_ws(s)}")

    sources = _extract_source_tags(list(episodic_notes) + list(extra_evidence_snippets))
    if not sources:
        for note in episodic_notes:
            sources.extend(_extract_source_tags([note]))
        sources = list(dict.fromkeys(sources))

    if sources:
        lines.append("")
        lines.append(f"Supporting sources (filenames cited in evidence): {'; '.join(sources[:12])}")

    return strip_user_facing_subquery_labels(normalize_ws("\n".join(lines)))
