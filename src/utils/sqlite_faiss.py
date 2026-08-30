"""Local FAISS retrieval from SQLite kvstore files.

Reads the OGX 1.0 kvstore layout (``vector_io::faiss:`` + ``v3`` keys,
``IndexFlatL2``) that rag-content writes. No ``ogx`` / ``llama_stack``
imports; only stdlib + numpy + faiss + sentence-transformers.
"""

import asyncio
import base64
import io
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Final

import faiss
import numpy as np

from log import get_logger

logger = get_logger(__name__)

KV_NAMESPACE: Final[str] = "vector_io::faiss"
KV_VERSION: Final[str] = "v3"


def _kv_key(kind: str, vector_store_id: str) -> str:
    """Build a store-level KV key."""
    return f"{KV_NAMESPACE}:{kind}:{KV_VERSION}::{vector_store_id}"


def _deserialize_faiss_index(payload: str) -> faiss.Index:
    """Decode an OGX-stored FAISS index blob (base64 of npy uint8 array)."""
    raw = base64.b64decode(payload)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return faiss.deserialize_index(arr)


@dataclass
class _ChunkResult:
    """Mimics the shape of an OGX vector_io.query response chunk."""

    content: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SearchResponse:
    """Mimics the shape of an OGX vector_io.query response."""

    chunks: list[_ChunkResult]
    scores: list[float]


@dataclass
class _CachedStore:
    """In-memory cache entry for a deserialized FAISS index and its chunks."""

    index: faiss.Index
    chunk_by_index: dict[str, str]


_store_cache: dict[tuple[str, str], _CachedStore] = {}
_store_cache_lock = asyncio.Lock()


async def _get_cached_store(db_path: str, vector_store_id: str) -> _CachedStore:
    """Return a cached FAISS index + chunk map, loading from SQLite on first access.

    The database file is read-only at serving time, so entries are cached
    unconditionally for the lifetime of the process.
    """
    cache_key = (db_path, vector_store_id)
    if cache_key in _store_cache:
        return _store_cache[cache_key]
    async with _store_cache_lock:
        if cache_key in _store_cache:
            return _store_cache[cache_key]
        store = await asyncio.to_thread(
            _load_store_from_sqlite, db_path, vector_store_id
        )
        _store_cache[cache_key] = store
        logger.info(
            "Cached FAISS index for %s (store=%s, vectors=%d)",
            db_path,
            vector_store_id,
            store.index.ntotal,
        )
    return _store_cache[cache_key]


def _load_store_from_sqlite(db_path: str, vector_store_id: str) -> _CachedStore:
    """Read and deserialize a FAISS store from the SQLite kvstore (blocking)."""
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT value FROM kvstore WHERE key=?",
            (_kv_key("faiss_index", vector_store_id),),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise KeyError(
            f"FAISS index not found for vector_store_id="
            f"{vector_store_id!r} in {db_path}"
        )

    payload = json.loads(row[0])
    index = _deserialize_faiss_index(payload["faiss_index"])
    return _CachedStore(index=index, chunk_by_index=payload["chunk_by_index"])


def _search_sqlite_faiss(  # pylint: disable=too-many-locals
    cached_store: _CachedStore,
    query_embedding: np.ndarray,
    k: int,
    score_threshold: float,
) -> _SearchResponse:
    """Search a cached FAISS index (blocking, CPU-bound).

    Parameters:
        cached_store: Pre-loaded FAISS index and chunk map.
        query_embedding: Query vector (1-D float32 array).
        k: Maximum number of nearest neighbors.
        score_threshold: Minimum similarity score to include.

    Returns:
        _SearchResponse with chunks and L2-distance-based scores.
    """
    index = cached_store.index
    query = query_embedding.reshape(1, -1).astype(np.float32)
    distances, labels = index.search(query, min(k, index.ntotal))

    chunks: list[_ChunkResult] = []
    scores: list[float] = []

    for distance, label in zip(distances[0], labels[0], strict=True):
        if int(label) < 0:
            continue
        similarity = 1.0 / (1.0 + float(distance))
        if similarity < score_threshold:
            continue
        chunk_data = json.loads(cached_store.chunk_by_index[str(int(label))])
        chunks.append(
            _ChunkResult(
                content=chunk_data["content"],
                chunk_id=chunk_data.get("chunk_id", ""),
                metadata=chunk_data.get("metadata", {}),
            )
        )
        scores.append(similarity)

    return _SearchResponse(chunks=chunks, scores=scores)


# Embedding model cache (same pattern as reranker.py)
_embedding_models: dict[str, Any] = {}
_embedding_model_lock = asyncio.Lock()


async def _get_embedding_model(model_name_or_path: str) -> Any:
    """Load or retrieve a cached SentenceTransformer model."""
    if model_name_or_path in _embedding_models:
        return _embedding_models[model_name_or_path]
    async with _embedding_model_lock:
        if model_name_or_path in _embedding_models:
            return _embedding_models[model_name_or_path]
        from sentence_transformers import (  # pylint: disable=import-outside-toplevel
            SentenceTransformer,
        )

        model = await asyncio.to_thread(SentenceTransformer, model_name_or_path)
        _embedding_models[model_name_or_path] = model
        logger.info("Loaded embedding model for local FAISS: %s", model_name_or_path)
    return _embedding_models[model_name_or_path]


async def query_sqlite_faiss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    db_path: str,
    embedding_model: str,
    query: str,
    max_chunks: int,
    score_threshold: float,
    vector_store_id: str,
) -> _SearchResponse:
    """Embed a query and search a local sqlite-faiss file.

    Runs the embedding and FAISS search in a thread pool to avoid blocking
    the event loop.

    Parameters:
        db_path: Path to the kvstore SQLite file.
        embedding_model: SentenceTransformer model name or local path.
        query: Search query string.
        max_chunks: Maximum number of results.
        score_threshold: Minimum similarity score to include.
        vector_store_id: Store id inside the SQLite file.

    Returns:
        _SearchResponse compatible with ``_extract_byok_rag_chunks``.
    """
    model = await _get_embedding_model(embedding_model)
    query_embedding = await asyncio.to_thread(model.encode, query)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    cached_store = await _get_cached_store(db_path, vector_store_id)

    return await asyncio.to_thread(
        _search_sqlite_faiss,
        cached_store,
        query_embedding,
        max_chunks,
        score_threshold,
    )


def list_vector_store_ids(db_path: str) -> list[str]:
    """Return vector_store_ids present in a sqlite-faiss SQLite file."""
    prefix = f"{KV_NAMESPACE}:faiss_index:{KV_VERSION}::"
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT key FROM kvstore WHERE key LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0][len(prefix) :]) for row in rows]
