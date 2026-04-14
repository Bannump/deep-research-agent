"""Local corpus chunking, Chroma indexing, and semantic retrieval."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from app.config import get_settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float  # higher is better (distance inverted)


class DocumentRetriever:
    """
    Indexes `data/sample_docs` into Chroma and retrieves top-k chunks per query.
    Uses Chroma default embedding function unless USE_MOCK_EMBEDDINGS=1 (tests).
    """

    COLLECTION_NAME = "g3_research_corpus"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._client: ClientAPI = chromadb.PersistentClient(
            path=self._settings.chroma_path.as_posix()
        )
        self._collection = self._get_or_create_collection()
        self._ensure_corpus_indexed()

    def _embedding_function(self) -> Any:
        if self._settings.use_mock_embeddings:
            from chromadb.utils.embedding_functions import EmbeddingFunction

            class MockEmbeddingFunction(EmbeddingFunction):  # type: ignore[misc]
                def __init__(self) -> None:
                    super().__init__()

                def name(self) -> str:
                    return "mock-sha256-32d"

                def __call__(self, input: list[str]) -> list[list[float]]:
                    out: list[list[float]] = []
                    for t in input:
                        h = hashlib.sha256(t.encode("utf-8")).digest()
                        vec = [((b - 128) / 128.0) for b in h[:32]]
                        out.append(vec)
                    return out

            return MockEmbeddingFunction()

        from chromadb.utils import embedding_functions

        return embedding_functions.DefaultEmbeddingFunction()

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function(),
        )

    def _read_corpus_files(self) -> list[tuple[str, str]]:
        root = self._settings.sample_docs_dir
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            return []
        texts: list[tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith((".txt", ".md")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        texts.append((os.path.relpath(path, str(root)), f.read()))
                except OSError:
                    continue
        return texts

    def _chunk_text(self, source: str, full_text: str) -> list[dict[str, Any]]:
        size = self._settings.chunk_size_chars
        overlap = self._settings.chunk_overlap_chars
        chunks: list[dict[str, Any]] = []
        t = full_text
        start = 0
        idx = 0
        while start < len(t):
            end = min(len(t), start + size)
            piece = t[start:end].strip()
            if piece:
                cid = hashlib.sha256(f"{source}:{idx}:{piece[:80]}".encode()).hexdigest()[:16]
                chunks.append(
                    {
                        "id": cid,
                        "document": piece,
                        "metadata": {"source": source, "chunk_index": idx},
                    }
                )
                idx += 1
            if end >= len(t):
                break
            start = max(0, end - overlap)
        return chunks

    def _ensure_corpus_indexed(self) -> None:
        if self._collection.count() > 0:
            return
        all_chunks: list[dict[str, Any]] = []
        for source, text in self._read_corpus_files():
            all_chunks.extend(self._chunk_text(source, text))
        if not all_chunks:
            # Minimal placeholder so retrieval never crashes on empty corpus
            all_chunks = [
                {
                    "id": "placeholder",
                    "document": "No sample documents found. Add .txt/.md files under data/sample_docs.",
                    "metadata": {"source": "system", "chunk_index": 0},
                }
            ]
        self._collection.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["document"] for c in all_chunks],
            metadatas=[c["metadata"] for c in all_chunks],
        )

    def retrieve(
        self,
        subquery: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self._settings.max_retrieved_chunks
        res = self._collection.query(query_texts=[subquery], n_results=k, include=["documents", "metadatas", "distances"])
        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[RetrievedChunk] = []
        for i, cid in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            source = str(meta.get("source", "unknown"))
            dist = float(dists[i]) if i < len(dists) else 1.0
            # Cosine distance: lower is better → score = 1 - d
            score = max(0.0, 1.0 - dist)
            out.append(
                RetrievedChunk(
                    chunk_id=str(cid),
                    source=source,
                    text=text or "",
                    score=score,
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out


_retriever_singleton: DocumentRetriever | None = None


def get_retriever() -> DocumentRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = DocumentRetriever()
    return _retriever_singleton
