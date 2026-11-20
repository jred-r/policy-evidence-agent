"""Validated data and answer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class Question(StrictModel):
    """A security assessment question."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: str = "general"


class PolicyPassage(StrictModel):
    """A source passage used to ground an answer."""

    id: str
    document_id: str
    text: str


class PolicyDocument(StrictModel):
    """A policy source with provenance."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    passages: list[PolicyPassage]


class AnswerKeyEntry(StrictModel):
    """Hand-authored benchmark answer and its supporting passages."""

    question_id: str
    answer: str
    supporting_passage_ids: list[str] = Field(min_length=1)


class Citation(StrictModel):
    """Citation to a retrieved source passage."""

    passage_id: str
    quote: str = ""


class DraftedAnswer(StrictModel):
    """Structured model output for one question."""

    question_id: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(min_length=1)


class DataSet(StrictModel):
    """Complete validated input corpus."""

    questions: list[Question]
    policies: list[PolicyDocument]
    answer_key: list[AnswerKeyEntry]


def load_data(data_dir: Path) -> DataSet:
    """Load and cross-validate the questionnaire corpus."""
    questions = _load_json(data_dir / "questions.json", list[Question])
    policies = [
        PolicyDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((data_dir / "policies").glob("*.json"))
    ]
    answer_key = _load_json(data_dir / "answer_key.json", list[AnswerKeyEntry])
    if not policies:
        raise ValueError(f"No policy JSON files found in {data_dir / 'policies'}")

    question_ids = [item.id for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question IDs must be unique")
    passage_ids = [passage.id for doc in policies for passage in doc.passages]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Policy passage IDs must be unique")
    known_questions = set(question_ids)
    known_passages = set(passage_ids)
    for entry in answer_key:
        if entry.question_id not in known_questions:
            raise ValueError(f"Unknown answer-key question: {entry.question_id}")
        if not set(entry.supporting_passage_ids) <= known_passages:
            raise ValueError(f"Unknown supporting passage for {entry.question_id}")
    if len(answer_key) != len({entry.question_id for entry in answer_key}):
        raise ValueError("Answer-key question IDs must be unique")
    if not 40 <= len(answer_key) <= 50:
        raise ValueError("The hand-authored answer key must contain 40-50 entries")
    return DataSet(questions=questions, policies=policies, answer_key=answer_key)


def _load_json(path: Path, model: Any) -> Any:
    """Load JSON and validate it against a Pydantic model."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(model).validate_python(raw)
