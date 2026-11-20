"""Single-pass policy-grounded conversational answering."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from ..data.retrieve import PolicyRetriever
from ..domain.schemas import Question
from ..llm.client import override_llm_mode
from .draft import contextualize_question, draft_answers_with_sources
from .scoring import grounding_f1

logger = logging.getLogger(__name__)
HUMAN_REVIEW_GROUNDING_THRESHOLD = 0.5
PENDING_REVIEW_ANSWER = "This answer is awaiting human review."


def answer_question(
    question_id: str,
    question: str,
    retriever: PolicyRetriever | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    llm_mode: Literal["record", "replay"] | None = None,
) -> dict[str, Any]:
    """Return one cited answer or a brief out-of-scope redirect."""
    question_id = question_id.strip()
    question = question.strip()
    if not question_id or not question:
        raise ValueError("question_id and question are required")

    history = conversation_history or []
    search = retriever or PolicyRetriever()
    draft_question = Question(
        id=question_id,
        text=contextualize_question(question, history),
    )
    started = time.monotonic()
    logger.info("answer_started question_id=%s", question_id)
    try:
        if llm_mode is None:
            answers, retrieved = draft_answers_with_sources(
                [draft_question], search, batch_size=1
            )
        else:
            with override_llm_mode(llm_mode):
                answers, retrieved = draft_answers_with_sources(
                    [draft_question], search, batch_size=1
                )
        answer = answers[0]
    except Exception as error:
        logger.error(
            "answer_failed question_id=%s error_type=%s",
            question_id,
            type(error).__name__,
        )
        raise

    grounding_score = None
    if answer.policy_related:
        source_text = {
            passage.passage_id: passage.text
            for passage in retrieved[question_id]
        }
        grounding_score = grounding_f1(answer, source_text)
    review_required = (
        answer.policy_related
        and grounding_score is not None
        and grounding_score < HUMAN_REVIEW_GROUNDING_THRESHOLD
    )
    status = "answered" if answer.in_scope else "off_topic"
    answer_payload = answer.model_dump()
    final_answer = answer_payload
    if review_required:
        final_answer = {
            **answer_payload,
            "answer": PENDING_REVIEW_ANSWER,
            "citations": [],
        }
    logger.info(
        "answer_completed question_id=%s status=%s citation_count=%s "
        "grounding_f1=%s review_required=%s duration_ms=%.2f",
        question_id,
        status,
        len(answer.citations),
        f"{grounding_score:.3f}" if grounding_score is not None else "n/a",
        review_required,
        (time.monotonic() - started) * 1000,
    )
    return {
        "question_id": question_id,
        "status": status,
        "final_answer": final_answer,
        "grounding_f1": grounding_score,
        "human_review_required": review_required,
        **({"review_draft": answer_payload} if review_required else {}),
    }
