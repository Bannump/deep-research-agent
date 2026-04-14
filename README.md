# Binox G3 — Deep Research Agent (Memory-Constrained Pipeline)

A **constrained research pipeline** (not a free-form chat agent) that answers multi-part questions under explicit **retrieval**, **working-memory**, and **episodic** limits. It logs what was retrieved, retained, summarized, and discarded, and exposes results through **FastAPI** with optional **n8n** orchestration.

## Why this architecture

- **Deterministic stages** (plan → retrieve → prune → answer → episodic compress → synthesize) are easier to debug and review than opaque agent loops.
- **Three memory tiers** separate *what the model sees now* (working memory), *what we keep as structured history* (SQLite episodic), and *what we can search again* (Chroma long-term evidence).
- **Explicit constraints** (chunk caps, token budget, subquery cap) make tradeoffs visible in API responses—important for enterprise and client research workflows.

## Memory tiers

| Tier | Role | Implementation |
|------|------|------------------|
| **Working memory** | Active LLM context per subquery | Top chunks after retrieval + scoring, capped at **4 chunks** and **~2000 estimated tokens** |
| **Episodic memory** | Compressed subquery results + session logs | **SQLite** via SQLAlchemy |
| **Long-term memory** | Raw evidence chunks + metadata for semantic recall | **ChromaDB** (persistent) |

## System architecture

- **FastAPI** (`app/main.py`) runs the pipeline and persists sessions.
- **Planner** (`app/planner.py`) emits up to **3** subqueries (LLM JSON with deterministic fallback).
- **Retriever** (`app/retriever.py`) chunks `data/sample_docs`, embeds (Chroma default or mock for tests), returns up to **8** candidates per subquery.
- **Memory manager** (`app/memory_manager.py`) enforces caps, assigns **discard reasons** (`low_relevance`, `duplicate`, `memory_limit`, `chunk_limit`), and writes episodic summaries.
- **Synthesizer** (`app/synthesizer.py`) answers subqueries from retained evidence and builds the final answer from episodic notes + short evidence excerpts.
- **Grounded synthesis** (`app/grounded_synthesis.py`) provides deterministic, extractive answers when `LLM_PROVIDER=mock` (or when API keys are missing so the mock provider is used): subquery answers and final synthesis quote actual retained text; subqueries are **sanitized** (prefix labels and duplicates removed).
- **LLM** (`app/llm.py`) uses a small **provider interface** (`mock` by default, optional OpenAI / Anthropic via env).

See [architecture.md](./architecture.md) for diagrams and data flow.

## Requirements

- Python **3.11+**
- Docker / Docker Compose (optional, for n8n + API bundle)

## Environment variables

Copy `.env.example` to `.env` and adjust:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `mock` (default), `openai`, or `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Required when using real providers |
| `WORKING_MEMORY_TOKEN_BUDGET` | Approximate token cap per subquery step (default `2000`) |
| `DEFAULT_COST_PER_1K_TOKENS_USD` | Rough cost model for reporting (default `0.002`) |
| `DATA_DIR`, `CHROMA_PATH`, `SQLITE_PATH`, `SAMPLE_DOCS_DIR` | Storage paths (Docker sets these automatically) |
| `USE_MOCK_EMBEDDINGS` | `1` for deterministic embeddings (tests / CI) |

Token counts are **approximate** (~4 characters per token); cost is **linear in estimated tokens** and intended for comparison, not billing.

## Run locally (Python)

From the project root (`binox-g3-research-agent`):

1. Create a venv and install dependencies:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Optional: `cp .env.example .env` (or `copy .env.example .env` on Windows) and edit variables. If you skip this, defaults apply (`LLM_PROVIDER=mock` is fine for the MVP).

3. Start the API (run from project root so `app` is importable):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First run indexes `data/sample_docs` into Chroma. With default settings, Chroma may download its **default embedding model** once (requires network the first time). For fully offline tests, set `USE_MOCK_EMBEDDINGS=1`.

**Adding or replacing corpus files:** Remove the persisted Chroma directory (`data/chroma_db` locally, or the Docker volume data) so the collection is rebuilt on next startup; otherwise old embeddings may linger.

## Run with Docker Compose

No `.env` file is required. Compose reads defaults from `docker-compose.yml`; optionally create `.env` in the same directory to set `API_PORT`, `N8N_PORT`, `LLM_PROVIDER`, or API keys, or run:

`docker compose --env-file .env up --build`

```bash
cd binox-g3-research-agent
docker compose up --build
```

- API: `http://localhost:8000` (host port from `API_PORT`, default `8000`)
- n8n UI: `http://localhost:5678` (host port from `N8N_PORT`, default `5678`)

The API container stores SQLite + Chroma in the `research_data` volume under `/data`.

## API usage

### `GET /health`

```bash
curl -s http://localhost:8000/health
```

### `POST /research`

**Windows (cmd)**

```bat
curl -s -X POST http://localhost:8000/research ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is the Aurora Compliance Framework? What technical controls are required for encryption and logging? What risks are called out?\"}"
```

**macOS / Linux / PowerShell (use a single line or backticks)**

```bash
curl -s -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Aurora Compliance Framework? What technical controls are required for encryption and logging? What risks are called out?"}'
```

The JSON is optimized for reviewers:

- **`reviewer_note`** — one paragraph orienting you to the run.
- **`constraints`** — active caps (subqueries, retrieval, working memory).
- **`subquery_results[]`** — per step: `retained_snippets`, typed **`discarded_chunks`** (reason codes), **`episodic_summary`**, `subquery_answer`.
- **`episodic_summaries`** — the episodic chain used toward synthesis.
- **`demonstration`** — short plain-language pointers for **memory pruning**, **episodic compression**, and **final synthesis**.
- **`session_metrics`** — totals plus **`discard_breakdown`** by reason.
- **`memory_actions`** — structured pipeline log.
- **`total_estimated_tokens` / `total_estimated_cost`** — duplicated from `session_metrics` for quick scanning.

### `GET /session/{session_id}`

Returns the persisted session (same structured fields).

### `GET /sessions?limit=20`

Recent sessions with **`query_preview`**, optional **`subquery_count`**, and a one-line **`one_line`** summary for logs or dashboards.

## Example: sample query and response shape (`LLM_PROVIDER=mock`)

**Request**

```json
{
  "query": "According to the Aurora Compliance Framework overview, what principles does ACF emphasize? Separately, what does the security review checklist require for penetration testing evidence? Finally, what availability target does the operating level agreement set for the production API?"
}
```

With **`LLM_PROVIDER=mock`**, decomposition still uses a small JSON template from the mock provider, but **answers are not placeholders**: they are **grounded extractive summaries** from retained chunks (plus structured final synthesis). The **structure** matches a real provider: retrieval counts, **`discarded_chunks`** with reasons, shorter **`episodic_summaries`**, and **`demonstration`** explaining the three memory stages.

**Abbreviated response (illustrative)**

```json
{
  "session_id": "…",
  "original_query": "…",
  "constraints": {
    "max_subqueries": 3,
    "max_retrieved_chunks_per_subquery": 8,
    "max_retained_chunks_working_memory": 4,
    "working_memory_token_budget_estimated": 2000
  },
  "subqueries": ["…", "…", "…"],
  "subquery_results": [
    {
      "subquery_index": 0,
      "question": "…",
      "retrieved_count": 8,
      "retained_count": 4,
      "discarded_count": 4,
      "retained_snippets": [{ "chunk_id": "…", "source": "01_overview.md", "text_preview": "…" }],
      "discarded_chunks": [{ "chunk_id": "…", "source": "…", "reason": "chunk_limit", "detail": "…" }],
      "subquery_answer": "From the retained excerpts… - … [01_overview.md]",
      "episodic_summary": "[What principles…] …",
      "working_memory_tokens": 123
    }
  ],
  "episodic_summaries": ["…", "…", "…"],
  "final_answer": "**Direct answer:** …\\n\\n**Supporting details…**",
  "session_metrics": {
    "subquery_count": 3,
    "chunks_retrieved_total": 24,
    "discard_breakdown": { "chunk_limit": 6, "duplicate": 2 }
  },
  "demonstration": {
    "memory_pruning": "Retrieval produced …",
    "episodic_compression": "Each subquery answer was compressed …",
    "final_synthesis": "The final answer was generated from …"
  },
  "reviewer_note": "Session `xxxxxxxx…`: 3 subqueries. Pruning: …"
}
```

Exact numbers depend on your corpus, Chroma scores, and token estimate. Run the same query twice with the same corpus and mock LLM for reproducible structure.

## n8n orchestration

Core logic stays in Python. **n8n** only triggers the API and returns JSON.

1. Import `workflows/g3_n8n_workflow.json` into n8n (Workflows → Import).
2. Ensure the HTTP Request URL matches your deployment:
   - Docker Compose (from n8n container): `http://api:8000/research` (default in file).
   - Host machine testing n8n outside Docker: `http://localhost:8000/research`.
3. Activate the workflow and POST to the webhook URL n8n shows, body: `{"query":"..."}`.

## Sample demo flow

1. Start the API (`LLM_PROVIDER=mock` is fine).
2. POST a multi-part question (see [Example](#example-sample-query-and-response-shape-llm_providermock)).
3. Read **`demonstration`** and **`subquery_results[].discarded_chunks`** to see pruning; read **`episodic_summaries`** then **`final_answer`** to trace episodic → synthesis.
4. `GET /session/{session_id}` to confirm SQLite persistence and the same structured fields.

## Project limitations (MVP)

- **Token estimates** are heuristic, not tokenizer-exact.
- **Corpus** is local static files under `data/sample_docs` (no live web retrieval in the default path).
- **Cost** figures are configurable approximations, not invoice-grade.
- **Embeddings**: Chroma’s default embedding may download ONNX/model assets on first run; use `USE_MOCK_EMBEDDINGS=1` for fully offline tests.

## Future improvements

- Pluggable **web retrieval** source behind the same retriever interface.
- Stronger **reranker** (cross-encoder) while keeping working-memory caps.
- **Tokenizer-based** token counting optional toggle.
- Optional **Redis** queue between n8n and API for load shedding.

## Manual setup checklist

1. **Python path**: Run `uvicorn` from the project root so `app` resolves as a package (or set `PYTHONPATH` to the project root).
2. **First-time Chroma embeddings**: With default settings (`USE_MOCK_EMBEDDINGS` unset), the first run may download embedding assets; needs network once unless you set `USE_MOCK_EMBEDDINGS=1` (deterministic, no download).
3. **Docker Compose**: No `.env` file is required; create one only to override ports or LLM keys. After `docker compose up`, import the n8n workflow and activate it (see [n8n orchestration](#n8n-orchestration)).
4. **Real LLM providers**: Set `LLM_PROVIDER` and the matching API key. If the provider is selected but the key is missing, the app **falls back to `MockLLMProvider`** so the API still starts.

## Evaluation writeup

See [evaluation.md](./evaluation.md).

## License

Prototype for assessment / internal use.
