"""Environment and runtime configuration for the research pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# Paths (override with env for Docker)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
SAMPLE_DOCS_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", str(DATA_DIR / "sample_docs")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db")))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(DATA_DIR / "sessions.db")))

# Pipeline constraints (spec)
MAX_SUBQUERIES = 3
MAX_RETRIEVED_CHUNKS = 8
MAX_RETAINED_CHUNKS = 4
WORKING_MEMORY_TOKEN_BUDGET = int(os.getenv("WORKING_MEMORY_TOKEN_BUDGET", "2000"))

# Chunking for local corpus
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))

# Cost / tokens (approximate; documented in evaluation.md)
DEFAULT_COST_PER_1K_TOKENS_USD = float(os.getenv("DEFAULT_COST_PER_1K_TOKENS_USD", "0.002"))

# LLM
LLM_PROVIDER: Literal["openai", "anthropic", "mock"] = os.getenv(  # type: ignore[assignment]
    "LLM_PROVIDER", "mock"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# Embeddings (Chroma default uses local model; set USE_MOCK_EMBEDDINGS=1 for deterministic tests)
USE_MOCK_EMBEDDINGS = os.getenv("USE_MOCK_EMBEDDINGS", "0") == "1"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


@lru_cache
def get_settings() -> "Settings":
    return Settings()


class Settings:
    """Typed accessor for config (cached singleton via get_settings)."""

    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.sample_docs_dir = SAMPLE_DOCS_DIR
        self.chroma_path = CHROMA_PATH
        self.sqlite_path = SQLITE_PATH
        self.max_subqueries = MAX_SUBQUERIES
        self.max_retrieved_chunks = MAX_RETRIEVED_CHUNKS
        self.max_retained_chunks = MAX_RETAINED_CHUNKS
        self.working_memory_token_budget = WORKING_MEMORY_TOKEN_BUDGET
        self.chunk_size_chars = CHUNK_SIZE_CHARS
        self.chunk_overlap_chars = CHUNK_OVERLAP_CHARS
        self.default_cost_per_1k_tokens_usd = DEFAULT_COST_PER_1K_TOKENS_USD
        self.llm_provider = LLM_PROVIDER
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.anthropic_api_key = ANTHROPIC_API_KEY
        self.anthropic_model = ANTHROPIC_MODEL
        self.use_mock_embeddings = USE_MOCK_EMBEDDINGS
        self.api_host = API_HOST
        self.api_port = API_PORT
