"""Tests for SqliteFaissSearchCapability."""

import json

import pytest
from pytest_mock import MockerFixture

from pydantic_ai_lightspeed.capabilities.sqlite_faiss_search import (
    KNOWLEDGE_SEARCH_TOOL_NAME,
    SqliteFaissSearchCapability,
    _search_all_stores,
)


class TestSqliteFaissSearchCapability:
    """Tests for the SqliteFaissSearchCapability dataclass."""

    def test_exposes_knowledge_search_toolset(self, mocker: MockerFixture) -> None:
        """Capability exposes a toolset with knowledge_search."""
        store = mocker.Mock()
        store.rag_id = "my-kb"
        store.vector_db_id = "vs-1"
        store.backend = "faiss"
        store.db_path = "/data/store.sqlite"
        store.embedding_model = "all-MiniLM-L6-v2"

        cap = SqliteFaissSearchCapability(stores=[store])
        toolset = cap.get_toolset()

        assert toolset is not None
        tool_defs = toolset.tools
        assert KNOWLEDGE_SEARCH_TOOL_NAME in tool_defs

    def test_no_serialization_name(self) -> None:
        """Capability has no serialization name (not spec-expressible)."""
        assert SqliteFaissSearchCapability.get_serialization_name() is None


class TestSearchAllStores:
    """Tests for _search_all_stores helper."""

    @pytest.mark.asyncio
    async def test_returns_ranked_results(self, mocker: MockerFixture) -> None:
        """Results from multiple stores are merged and ranked by score."""
        store1 = mocker.Mock()
        store1.rag_id = "kb-a"
        store1.vector_db_id = "vs-a"
        store1.db_path = "/data/a.sqlite"
        store1.embedding_model = "model-a"

        store2 = mocker.Mock()
        store2.rag_id = "kb-b"
        store2.vector_db_id = "vs-b"
        store2.db_path = "/data/b.sqlite"
        store2.embedding_model = "model-b"

        chunk_a = mocker.Mock(content="from A", metadata={"title": "A"})
        chunk_b = mocker.Mock(content="from B", metadata={"title": "B"})

        mock_query = mocker.patch(
            "pydantic_ai_lightspeed.capabilities.sqlite_faiss_search.query_sqlite_faiss",
            new_callable=mocker.AsyncMock,
        )
        mock_query.side_effect = [
            mocker.Mock(chunks=[chunk_a], scores=[0.7]),
            mocker.Mock(chunks=[chunk_b], scores=[0.9]),
        ]

        results = await _search_all_stores(
            [store1, store2], "test query", max_chunks=10, score_threshold=0.3
        )

        assert len(results) == 2
        assert results[0]["content"] == "from B"
        assert results[0]["score"] == 0.9
        assert results[0]["source"] == "kb-b"
        assert results[1]["content"] == "from A"
        assert results[1]["score"] == 0.7

    @pytest.mark.asyncio
    async def test_respects_max_chunks(self, mocker: MockerFixture) -> None:
        """Only top max_chunks results are returned."""
        store = mocker.Mock()
        store.rag_id = "kb"
        store.vector_db_id = "vs"
        store.db_path = "/data/store.sqlite"
        store.embedding_model = "model"

        chunks = [mocker.Mock(content=f"c{i}", metadata={}) for i in range(5)]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]

        mock_query = mocker.patch(
            "pydantic_ai_lightspeed.capabilities.sqlite_faiss_search.query_sqlite_faiss",
            new_callable=mocker.AsyncMock,
        )
        mock_query.return_value = mocker.Mock(chunks=chunks, scores=scores)

        results = await _search_all_stores(
            [store], "query", max_chunks=3, score_threshold=0.3
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_handles_store_failure_gracefully(
        self, mocker: MockerFixture
    ) -> None:
        """A failing store does not prevent results from other stores."""
        store_good = mocker.Mock()
        store_good.rag_id = "kb-good"
        store_good.vector_db_id = "vs-good"
        store_good.db_path = "/data/good.sqlite"
        store_good.embedding_model = "model"

        store_bad = mocker.Mock()
        store_bad.rag_id = "kb-bad"
        store_bad.vector_db_id = "vs-bad"
        store_bad.db_path = "/data/bad.sqlite"
        store_bad.embedding_model = "model"

        chunk = mocker.Mock(content="hello", metadata={})

        mock_query = mocker.patch(
            "pydantic_ai_lightspeed.capabilities.sqlite_faiss_search.query_sqlite_faiss",
            new_callable=mocker.AsyncMock,
        )
        mock_query.side_effect = [
            RuntimeError("connection failed"),
            mocker.Mock(chunks=[chunk], scores=[0.8]),
        ]

        results = await _search_all_stores(
            [store_bad, store_good], "query", max_chunks=10, score_threshold=0.3
        )

        assert len(results) == 1
        assert results[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_tool_returns_json(self, mocker: MockerFixture) -> None:
        """The knowledge_search tool function returns valid JSON."""
        store = mocker.Mock()
        store.rag_id = "kb"
        store.vector_db_id = "vs"
        store.db_path = "/data/store.sqlite"
        store.embedding_model = "model"

        chunk = mocker.Mock(content="result text", metadata={"title": "T"})

        mock_query = mocker.patch(
            "pydantic_ai_lightspeed.capabilities.sqlite_faiss_search.query_sqlite_faiss",
            new_callable=mocker.AsyncMock,
        )
        mock_query.return_value = mocker.Mock(chunks=[chunk], scores=[0.85])

        cap = SqliteFaissSearchCapability(stores=[store])
        toolset = cap.get_toolset()
        assert toolset is not None

        tool = toolset.tools[KNOWLEDGE_SEARCH_TOOL_NAME]
        result_str = await tool.function(query="test")
        parsed = json.loads(result_str)

        assert len(parsed) == 1
        assert parsed[0]["content"] == "result text"
        assert parsed[0]["score"] == 0.85
