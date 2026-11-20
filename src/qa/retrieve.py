"""Local policy ingestion and Chroma retrieval."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schemas import PolicyDocument


class Collection(Protocol):
    """Chroma collection operations used by the retriever."""

    def upsert(self, **kwargs: object) -> None: ...

    def query(self, **kwargs: object) -> dict[str, object]: ...


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
        persist_dir: Path | None = None,
        embedding_model: str | None = None,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        collection: Collection | None = None,
    ) -> None:
        self._embedder = embedder or _sentence_embedder(
            embedding_model or os.getenv("QA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        if collection is None:
            import chromadb

            client = chromadb.PersistentClient(
                path=str(persist_dir or os.getenv("QA_CHROMA_PATH", ".qa/chroma"))
            )
            collection = client.get_or_create_collection(
                name="policy_passages", metadata={"hnsw:space": "cosine"}
            )
        self._collection = collection

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
