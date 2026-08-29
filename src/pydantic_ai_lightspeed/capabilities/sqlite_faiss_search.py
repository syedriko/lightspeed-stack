"""Capability that exposes local sqlite-faiss stores as a function tool.

When added to an agent, the LLM can call ``knowledge_search(query)`` to
search local FAISS stores on demand — the same tool-RAG semantics as
``file_search`` but without an OGX round-trip.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

import constants
from log import get_logger
from models.config import RagStore
from utils.sqlite_faiss import query_sqlite_faiss

logger = get_logger(__name__)

KNOWLEDGE_SEARCH_TOOL_NAME: Final[str] = "knowledge_search"


@dataclass
class SqliteFaissSearchCapability(AbstractCapability[object]):
    """Provide a ``knowledge_search`` function tool backed by local sqlite-faiss stores."""

    stores: list[RagStore]
    max_chunks: int = constants.DEFAULT_TOOL_RAG_MAX_CHUNKS
    score_threshold: float = constants.DEFAULT_BYOK_RAG_RELEVANCE_CUTOFF_SCORE

    _toolset: FunctionToolset[object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the function toolset with a knowledge_search tool."""
        ts: FunctionToolset[object] = FunctionToolset()

        stores = self.stores
        max_chunks = self.max_chunks
        score_threshold = self.score_threshold

        @ts.tool_plain
        async def knowledge_search(query: str) -> str:
            """Search internal knowledge bases for information relevant to the query.

            Use this tool when you need to find documentation, procedures, or
            technical information to answer the user's question.

            Args:
                query: The search query describing what information you need.

            Returns:
                JSON array of matching chunks with content, score, and metadata.
            """
            results = await _search_all_stores(
                stores, query, max_chunks, score_threshold
            )
            return json.dumps(results, ensure_ascii=False)

        self._toolset = ts

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return None; this capability is not expressible via the spec."""
        return None

    def get_toolset(self) -> FunctionToolset[object] | None:
        """Return the toolset containing the knowledge_search tool."""
        return self._toolset


async def _search_all_stores(
    stores: list[RagStore],
    query: str,
    max_chunks: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Search all configured local FAISS stores and merge ranked results."""
    search_tasks = [
        query_sqlite_faiss(
            db_path=store.db_path,  # type: ignore[arg-type]
            embedding_model=store.embedding_model,
            query=query,
            max_chunks=max_chunks,
            score_threshold=score_threshold,
            vector_store_id=store.vector_db_id,
        )
        for store in stores
    ]

    responses = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_results: list[dict[str, Any]] = []
    for store, response in zip(stores, responses, strict=True):
        if isinstance(response, BaseException):
            logger.warning(
                "knowledge_search: failed to query '%s': %s",
                store.vector_db_id,
                response,
            )
            continue
        for chunk, score in zip(response.chunks, response.scores, strict=True):
            all_results.append(
                {
                    "content": chunk.content,
                    "score": score,
                    "source": store.rag_id,
                    "metadata": chunk.metadata,
                }
            )

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:max_chunks]
