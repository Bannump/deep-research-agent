# Evaluation — Memory-Constrained Deep Research Pipeline

## Purpose of this submission

This artifact demonstrates a **bounded, inspectable research workflow**: decomposition, retrieval, **working-memory pruning**, **episodic compression**, and **final synthesis**, with explicit accounting of what entered each stage. The design prioritizes **reproducibility and auditability** over open-ended agent autonomy—appropriate when research outputs must be explained to stakeholders or regulators.

## Engineering objective

Deliver a service that:

1. Accepts a multi-part question and **plans** a small number of focused subqueries (≤3).
2. **Retrieves** a capped set of evidence chunks per subquery, then **prunes** to a stricter working-memory budget (chunk count + approximate tokens).
3. **Summarizes** each subquery outcome into episodic notes suitable for cross-step reasoning without carrying full text forward.
4. **Synthesizes** a final answer from episodic notes plus a limited set of evidence excerpts.
5. Persists enough structure (SQLite + Chroma) to **reconstruct** decisions after the fact.

## Constraints and why they exist

| Parameter | Setting | Role in the system |
|-----------|---------|-------------------|
| Subqueries | ≤3 | Limits combinatorial explosion and keeps the plan legible. |
| Candidates / subquery | ≤8 | Bounds retrieval fan-out before scoring and pruning. |
| Working-memory chunks | ≤4 | Forces prioritization; mirrors “only the best sources on the desk.” |
| Working-memory tokens (est.) | ~2000 | Coarse guardrail aligned with a compact context window slice. |
| Cost model | Configurable $/1k est. tokens | Enables comparative reporting, not financial precision. |

These are **policy knobs**: they encode how aggressively the system trades completeness for controllability.

## Memory model (three tiers)

1. **Working memory** — Text passed to the LLM for the *current* subquery only, after retrieval and pruning. This is where **memory pruning** is visible: each discarded chunk carries a **reason code** (`low_relevance`, `duplicate`, `memory_limit`, `chunk_limit`).

2. **Episodic memory** — Short notes persisted per subquery (`episodic_summary` / `episodic_summaries`). They intentionally **lose verbatim detail** to prevent unbounded context growth while preserving **claims and themes** for synthesis.

3. **Long-term memory (vector store)** — Chunked corpus in Chroma for semantic retrieval. This is **not** the same as working memory: it is the pool from which candidates are drawn each step.

## Retrieval approach

The MVP uses a **local, static corpus** under `data/sample_docs`. Chroma provides **dense retrieval** with distance-based scoring; scores are mapped to a higher-is-better relevance score before pruning. **No live web** in the default path—runs are repeatable and failures are easier to reason about than with crawling dependencies.

**Operational note:** If you add or replace corpus files, delete the Chroma persistence directory (or volume) so the collection is rebuilt; otherwise stale embeddings may persist.

## Tradeoffs (frank)

| Tradeoff | Decision | Consequence |
|----------|----------|-------------|
| Detail vs. capacity | Hard caps on chunks/tokens | Some relevant passages are dropped; mitigated by logged discard reasons and multi-step coverage via subqueries. |
| Episodic vs. raw | Compress after each subquery | Faster synthesis and lower drift risk; small risk of nuance loss—mitigated by retaining short excerpts for final fusion. |
| Static corpus vs. live web | Static for MVP | High reproducibility; answers cannot reflect post-cutoff external reality without an extension. |
| Estimated vs. exact tokens | Character-based heuristic | Cheap and portable; not tokenizer-identical to any vendor API. |

## Failure modes we expect

- **Corpus mismatch:** Subqueries may not hit the right lexicon; decomposition and synonym-rich subqueries help.
- **Over-dedup:** Near-duplicate normalization may merge distinct lines; thresholds favor avoiding redundant context.
- **Embedding cold start:** First Chroma run may download model assets unless `USE_MOCK_EMBEDDINGS=1` is used (tests/offline).

## Why this is credible for enterprise-style research

- **Traceability:** Structured API responses and SQLite persistence show **what was considered**, **what was kept**, and **why something was discarded**.
- **Governance alignment:** Explicit limits map to proportionality and minimization narratives common in regulated environments.
- **Separation of orchestration and logic:** n8n can trigger runs without embedding research rules in low-code, keeping the core **reviewable in Python**.

## Future work (concise)

- Hybrid sparse + dense retrieval; optional cross-encoder reranking with the same downstream caps.
- Namespace-aware corpora per tenant; optional async job API with webhooks.
- Optional tokenizer-based token accounting behind the same interfaces.
