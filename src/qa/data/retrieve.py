"""Local policy ingestion and Chroma retrieval."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from math import fsum, sqrt
from pathlib import Path
from typing import Protocol

from psycopg.types.json import Jsonb

from ..domain.schemas import PolicyDocument
from ..storage.db import connection


class Collection(Protocol):
    """Passage storage operations used by the retriever."""

    def upsert(self, **kwargs: object) -> None: ...

    def query(self, **kwargs: object) -> dict[str, object]: ...


class _PostgresCollection:
    """PostgreSQL-backed passage store with cosine-distance ranking."""

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("Passage IDs, documents, metadata, and embeddings must align")
        rows = [
            (
                ids[index],
                metadatas[index]["document_id"],
                documents[index],
                Jsonb(embeddings[index]),
            )
            for index in range(len(ids))
        ]
        with connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO qa_policy_passages
                        (passage_id, document_id, passage_text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (passage_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        passage_text = EXCLUDED.passage_text,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    rows,
                )

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, object]:
        if len(query_embeddings) != 1:
            raise ValueError("Exactly one query embedding is required")
        query_vector = query_embeddings[0]
        if not query_vector or n_results < 1:
            raise ValueError("A non-empty query embedding and positive result count are required")

        with connection() as conn:
            rows = conn.execute(
                """
                SELECT passage_id, document_id, passage_text, embedding
                FROM qa_policy_passages
                """
            ).fetchall()

        query_norm = sqrt(fsum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("Query embedding has zero length")
        ranked: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            vector = row["embedding"]
            if len(vector) != len(query_vector):
                raise ValueError(
                    "Stored passage embeddings do not match the query embedding dimensions"
                )
            vector_norm = sqrt(fsum(value * value for value in vector))
            if vector_norm == 0:
                raise ValueError(f"Passage {row['passage_id']} has a zero-length embedding")
            similarity = fsum(
                left * right for left, right in zip(query_vector, vector, strict=True)
            ) / (query_norm * vector_norm)
            ranked.append((1.0 - similarity, row))
        ranked.sort(key=lambda item: item[0])
        selected = ranked[:n_results]
        return {
            "ids": [[row["passage_id"] for _distance, row in selected]],
            "documents": [[row["passage_text"] for _distance, row in selected]],
            "metadatas": [
                [{"document_id": row["document_id"]} for _distance, row in selected]
            ],
            "distances": [[distance for distance, _row in selected]],
        }


@dataclass(frozen=True)
class RetrievedPassage:
    """A passage returned by semantic search."""

    passage_id: str
    document_id: str
    text: str
    distance: float


class PolicyRetriever:
    """Store and search policy passages with local sentence embeddings."""

    def __init__(
        self,
        embedding_model: str | None = None,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        collection: Collection | None = None,
    ) -> None:
        self._embedder = embedder or _sentence_embedder(
            embedding_model or os.getenv("QA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self._collection = collection if collection is not None else _PostgresCollection()

    def ingest(self, documents: list[PolicyDocument]) -> int:
        """Upsert all policy passages and return their count."""
        passages = [passage for document in documents for passage in document.passages]
        if not passages:
            return 0
        texts = [passage.text for passage in passages]
        self._collection.upsert(
            ids=[passage.id for passage in passages],
            documents=texts,
            metadatas=[
                {"document_id": passage.document_id} for passage in passages
            ],
            embeddings=self._embed(texts),
        )
        return len(passages)

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedPassage]:
        """Return the closest policy passages for a question."""
        if k < 1:
            raise ValueError("k must be positive")
        vectors = self._embed([question])
        result = self._collection.query(
            query_embeddings=vectors,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first(result, "ids")
        documents = _first(result, "documents")
        metadatas = _first(result, "metadatas")
        distances = _first(result, "distances")
        return [
            RetrievedPassage(
                passage_id=str(ids[index]),
                document_id=str(metadatas[index]["document_id"]),
                text=str(documents[index]),
                distance=float(distances[index]),
            )
            for index in range(len(ids))
        ]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder(texts)


def _sentence_embedder(
    model_name: str,
) -> Callable[[list[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


def _first(result: dict[str, object], key: str) -> list[object]:
    values = result.get(key)
    if not isinstance(values, list) or not values:
        return []
    first = values[0]
    return first if isinstance(first, list) else []


def recall_at_k(
    retriever: PolicyRetriever,
    questions: list[tuple[str, set[str]]],
    k: int = 5,
) -> float:
    """Compute mean supporting-passage recall across questions."""
    if not questions:
        return 0.0
    scores = []
    for text, expected in questions:
        found = {item.passage_id for item in retriever.retrieve(text, k)}
        scores.append(len(found & expected) / len(expected) if expected else 0.0)
    return sum(scores) / len(scores)


def load_policy_documents(paths: list[Path]) -> list[PolicyDocument]:
    """Load JSON policy documents from disk."""

    return [
        PolicyDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(paths)
    ]
