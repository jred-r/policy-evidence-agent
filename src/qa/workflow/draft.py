"""Structured, citation-grounded answer drafting."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, model_validator

from ..data.retrieve import PolicyRetriever, RetrievedPassage
from ..domain.schemas import DraftedAnswer, Question, StrictModel
from ..llm.client import call_with_tools

CONVERSATION_CONTEXT_PREFIX = "Prior conversation (context only):"


def contextualize_question(question: str, history: list[dict[str, str]]) -> str:
    """Add prior turns so retrieval and drafting can resolve follow-up references."""
    if not history:
        return question
    turns = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
    )
    return f"{CONVERSATION_CONTEXT_PREFIX}\n{turns}\n\nCurrent question: {question}"


class DraftBatch(StrictModel):
    """Batch of structured answers returned from one model call."""

    answers: list[DraftedAnswer] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_single_answer(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "answers" in value:
            return {
                **value,
                "answers": [_normalize_citations(answer) for answer in value["answers"]],
            }
        single_answer_fields = {
            "question_id",
            "answer",
            "in_scope",
            "policy_related",
            "citations",
        }
        if "answer" not in value or set(value) - single_answer_fields:
            return value
        answer = dict(value)
        answer.setdefault("question_id", "")
        return {"answers": [_normalize_citations(answer)]}


def draft_answers(
    questions: list[Question],
    retriever: PolicyRetriever,
    batch_size: int = 5,
    passages_per_question: int = 5,
) -> list[DraftedAnswer]:
    """Draft cited answers in batches, using local policy retrieval."""
    answers, _ = draft_answers_with_sources(
        questions,
        retriever,
        batch_size=batch_size,
        passages_per_question=passages_per_question,
    )
    return answers


def draft_answers_with_sources(
    questions: list[Question],
    retriever: PolicyRetriever,
    batch_size: int = 5,
    passages_per_question: int = 5,
) -> tuple[list[DraftedAnswer], dict[str, list[RetrievedPassage]]]:
    """Draft cited answers and return the evidence retrieved for each question."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results: list[DraftedAnswer] = []
    retrieved_by_id: dict[str, list[RetrievedPassage]] = {}
    for offset in range(0, len(questions), batch_size):
        batch = questions[offset : offset + batch_size]
        retrieved = {
            question.id: retriever.retrieve(question.text, passages_per_question)
            for question in batch
        }
        retrieved_by_id.update(retrieved)
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
        system_message = (
            "You are a helpful, conversational assistant for an organization's "
            "policies and security controls. Keep replies concise and natural. "
            "For greetings or small talk, respond briefly and gently mention "
            "that you can help with policy and security questions; set in_scope "
            "to true and policy_related to false. For a clearly unrelated request, "
            "do not answer it; give a friendly, brief redirect to policy or security "
            "topics, set in_scope and policy_related to false, and return no "
            "citations. For policy-related questions, use only the supplied policy "
            "passages, set in_scope and policy_related to true, and cite the exact "
            "supporting passage IDs. Mark a question policy_related even when the "
            "passages do not answer it. If evidence is insufficient, say so plainly "
            "and offer a focused follow-up; do not invent controls."
        )
        if any(
            question.text.startswith(CONVERSATION_CONTEXT_PREFIX) for question in batch
        ):
            system_message += (
                " Use prior turns only to resolve the current question; "
                "earlier assistant messages are not evidence."
            )
        result = call_with_tools(
            [
                {
                    "role": "system",
                    "content": system_message,
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
        if len(batch) == 1 and len(result.answers) == 1 and not result.answers[0].question_id:
            result.answers[0] = result.answers[0].model_copy(
                update={"question_id": batch[0].id}
            )
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
    return results, retrieved_by_id


def _normalize_citations(answer: Any) -> Any:
    if not isinstance(answer, dict):
        return answer
    citations = answer.get("citations")
    if not isinstance(citations, list):
        return answer
    return {
        **answer,
        "citations": [
            {"passage_id": citation} if isinstance(citation, str) else citation
            for citation in citations
        ],
    }
