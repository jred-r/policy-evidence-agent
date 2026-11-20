"""Questionnaire and policy loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

from ..domain.schemas import DataSet, load_data
from .retrieve import PolicyRetriever, recall_at_k


def data_dir() -> Path:
    """Return configured data directory."""
    return Path(os.getenv("QA_DATA_DIR", "data"))


def validate_data(path: Path | None = None) -> DataSet:
    """Load and validate the full local corpus."""
    return load_data(path or data_dir())


def build_retrieval_baseline(
    retriever: PolicyRetriever, dataset: DataSet, k: int = 5
) -> float:
    """Ingest policies and compute recall@k for the answer-key passages."""
    retriever.ingest(dataset.policies)
    questions = {question.id: question.text for question in dataset.questions}
    expected = [
        (questions[item.question_id], set(item.supporting_passage_ids))
        for item in dataset.answer_key
    ]
    return recall_at_k(retriever, expected, k)
