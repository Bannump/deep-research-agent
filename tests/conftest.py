"""Pytest fixtures: isolated data dirs, mock embeddings, reset singletons."""

from __future__ import annotations

import os

# Ensure mock-friendly defaults before any test module imports `app`.
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "1")
os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "data"
    base.mkdir()
    (base / "sample_docs").mkdir()
    (base / "sample_docs" / "doc.txt").write_text(
        ("Aurora Compliance Framework emphasizes audit trails and least privilege. " * 30).strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("DATA_DIR", str(base))
    monkeypatch.setenv("CHROMA_PATH", str(base / "chroma"))
    monkeypatch.setenv("SQLITE_PATH", str(base / "sessions.db"))
    monkeypatch.setenv("SAMPLE_DOCS_DIR", str(base / "sample_docs"))

    import app.config as cfg
    import app.db as db
    import app.retriever as ret

    cfg.get_settings.cache_clear()
    db._engine = None
    db._engine_url = None
    ret._retriever_singleton = None
