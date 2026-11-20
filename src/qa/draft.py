"""Structured, citation-grounded answer drafting."""

from __future__ import annotations

import json

from pydantic import Field

from .llm import call_with_tools
from .retrieve import PolicyRetriever
from .schemas import DraftedAnswer, Question, StrictModel


class DraftBatch(StrictModel):
    """Batch of structured answers returned from one model call."""

    answers: list[DraftedAnswer] = Field(min_length=1)


def draft_answers(
    questions: list[Question],
    retriever: PolicyRetriever,
    batch_size: int = 5,
    passages_per_question: int = 5,
) -> list[DraftedAnswer]:
    """Draft cited answers in batches, using local policy retrieval."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results: list[DraftedAnswer] = []
    for offset in range(0, len(questions), batch_size):
        batch = questions[offset : offset + batch_size]
        retrieved = {
            question.id: retriever.retrieve(question.text, passages_per_question)
            for question in batch
        }
        sources = [
            {
                "question_id": question.id,
                "question": question.text,
                "passages": [
                    {
                        "passage_id": passage.passage_id,
                        "document_id": passage.document_id,
                        "text": passage.text,
                    }
                    for passage in retrieved[question.id]
                ],
            }
            for question in batch
        ]
        tool = {
            "type": "function",
            "function": {
                "name": "submit_answers",
                "description": "Submit a concise answer and cite supporting passage IDs.",
                "parameters": DraftBatch.model_json_schema(),
            },
        }
        result = call_with_tools(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer only from supplied policy passages. Do not infer "
                        "unstated controls. Cite one or more exact passage IDs. "
                        "Set confidence between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(sources, ensure_ascii=False, sort_keys=True),
                },
            ],
            [tool],
            DraftBatch,
        )
        questions_by_id = {question.id: question for question in batch}
        for answer in result.answers:
            question = questions_by_id.get(answer.question_id)
            if question is None:
                raise ValueError(f"Model returned unknown question {answer.question_id}")
            allowed_ids = {
                passage.passage_id for passage in retrieved[question.id]
            }
            if any(item.passage_id not in allowed_ids for item in answer.citations):
                raise ValueError(
                    f"Model cited a passage not supplied in batch: {answer.question_id}"
                )
        received = {answer.question_id for answer in result.answers}
        missing = {question.id for question in batch} - received
        if missing:
            raise ValueError(f"Model omitted answers for questions: {sorted(missing)}")
        results.extend(result.answers)
    return results
