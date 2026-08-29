"""Unit tests for the local sqlite-faiss retrieval module."""

# pylint: disable=too-few-public-methods

import base64
import io
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest
from pytest_mock import MockerFixture

from utils.sqlite_faiss import (
    KV_NAMESPACE,
    KV_VERSION,
    _kv_key,
    _load_store_from_sqlite,
    _search_sqlite_faiss,
    list_vector_store_ids,
    query_sqlite_faiss,
)


def _make_faiss_payload(embeddings: np.ndarray, chunks: list[dict]) -> dict:
    """Build a faiss_index payload like rag-content writes."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # pylint: disable=no-value-for-parameter

    buf = io.BytesIO()
    np.save(buf, faiss.serialize_index(index), allow_pickle=False)
    faiss_blob = base64.b64encode(buf.getvalue()).decode("ascii")

    chunk_by_index = {
        str(i): json.dumps(chunk, separators=(",", ":"))
        for i, chunk in enumerate(chunks)
    }
    return {"faiss_index": faiss_blob, "chunk_by_index": chunk_by_index}


def _create_test_db(
    tmp_path: Path, vector_store_id: str = "vs_test"
) -> tuple[str, np.ndarray]:
    """Create a test SQLite kvstore file with 3 chunks."""
    dim = 8
    embeddings = np.random.default_rng(42).random((3, dim)).astype(np.float32)
    chunks = [
        {
            "content": f"chunk {i}",
            "chunk_id": f"id_{i}",
            "metadata": {"title": f"doc_{i}", "document_id": f"file_{i}"},
        }
        for i in range(3)
    ]

    payload = _make_faiss_payload(embeddings, chunks)
    db_path = str(tmp_path / "faiss_store.db")
    key = f"{KV_NAMESPACE}:faiss_index:{KV_VERSION}::{vector_store_id}"

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kvstore "
        "(key TEXT PRIMARY KEY, value TEXT, expiration TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO kvstore (key, value, expiration) VALUES (?, ?, NULL)",
        (key, json.dumps(payload, separators=(",", ":"))),
    )
    conn.commit()
    conn.close()

    return db_path, embeddings


class TestKvKey:
    """Tests for _kv_key helper."""

    def test_builds_namespaced_key(self) -> None:
        """Produces the expected namespace:kind:version::store_id format."""
        result = _kv_key("faiss_index", "vs_abc")
        assert result == "vector_io::faiss:faiss_index:v3::vs_abc"


class TestSearchSqliteFaiss:
    """Tests for _search_sqlite_faiss."""

    def test_returns_nearest_chunk(self, tmp_path: Path) -> None:
        """Closest embedding is returned first."""
        db_path, embeddings = _create_test_db(tmp_path)
        cached = _load_store_from_sqlite(db_path, "vs_test")
        response = _search_sqlite_faiss(cached, embeddings[0], k=1, score_threshold=0.0)
        assert len(response.chunks) == 1
        assert response.chunks[0].content == "chunk 0"
        assert response.scores[0] == pytest.approx(1.0, abs=0.01)

    def test_respects_k_limit(self, tmp_path: Path) -> None:
        """At most k chunks are returned."""
        db_path, embeddings = _create_test_db(tmp_path)
        cached = _load_store_from_sqlite(db_path, "vs_test")
        response = _search_sqlite_faiss(cached, embeddings[0], k=2, score_threshold=0.0)
        assert len(response.chunks) == 2

    def test_respects_score_threshold(self, tmp_path: Path) -> None:
        """Chunks below the threshold are excluded."""
        db_path, embeddings = _create_test_db(tmp_path)
        cached = _load_store_from_sqlite(db_path, "vs_test")
        response = _search_sqlite_faiss(cached, embeddings[0], k=3, score_threshold=1.0)
        assert len(response.chunks) == 1
        assert response.chunks[0].content == "chunk 0"

    def test_raises_on_missing_key(self, tmp_path: Path) -> None:
        """KeyError raised when vector_store_id is absent from the DB."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE kvstore (key TEXT PRIMARY KEY, value TEXT, expiration TIMESTAMP)"
        )
        conn.commit()
        conn.close()

        with pytest.raises(KeyError, match="FAISS index not found"):
            _load_store_from_sqlite(db_path, "vs_missing")


class TestListVectorStoreIds:
    """Tests for list_vector_store_ids."""

    def test_lists_store_ids(self, tmp_path: Path) -> None:
        """All vector_store_ids present in the DB are returned."""
        db_path, _ = _create_test_db(tmp_path, "vs_one")
        # Add a second store key
        conn = sqlite3.connect(db_path)
        key2 = f"{KV_NAMESPACE}:faiss_index:{KV_VERSION}::vs_two"
        conn.execute(
            "INSERT INTO kvstore (key, value, expiration) VALUES (?, ?, NULL)",
            (key2, "{}"),
        )
        conn.commit()
        conn.close()

        ids = list_vector_store_ids(db_path)
        assert sorted(ids) == ["vs_one", "vs_two"]


class TestQuerySqliteFaiss:
    """Tests for the async query_sqlite_faiss function."""

    @pytest.mark.asyncio
    async def test_embeds_and_searches(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Embeds the query and delegates to the synchronous search."""
        db_path, embeddings = _create_test_db(tmp_path)

        mock_model = mocker.MagicMock()
        mock_model.encode.return_value = embeddings[0]

        async def _fake_get_model(_name: str) -> object:
            return mock_model

        mocker.patch(
            "utils.sqlite_faiss._get_embedding_model", side_effect=_fake_get_model
        )
        # Clear the store cache so the test DB is loaded fresh
        mocker.patch.dict("utils.sqlite_faiss._store_cache", clear=True)
        response = await query_sqlite_faiss(
            db_path=db_path,
            embedding_model="test-model",
            query="test query",
            max_chunks=1,
            score_threshold=0.0,
            vector_store_id="vs_test",
        )

        assert len(response.chunks) == 1
        assert response.chunks[0].content == "chunk 0"
