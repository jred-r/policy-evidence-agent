"""Runtime grounding and offline benchmark metrics."""

from __future__ import annotations

import re
from collections import Counter

from .schemas import AnswerKeyEntry, Citation, DraftedAnswer


def token_f1(answer: str, evidence: str) -> float:
    """Return multiset token F1 between a draft and cited evidence."""
    answer_tokens = Counter(re.findall(r"\w+", answer.casefold()))
    evidence_tokens = Counter(re.findall(r"\w+", evidence.casefold()))
    answer_count = sum(answer_tokens.values())
    evidence_count = sum(evidence_tokens.values())
    if not answer_count or not evidence_count:
        return 0.0
    overlap = sum((answer_tokens & evidence_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / answer_count
    recall = overlap / evidence_count
    return 2 * precision * recall / (precision + recall)


def grounding_f1(
    answer: DraftedAnswer, source_passages: dict[str, str]
) -> float:
    """Score answer-token overlap against its cited source passages."""
    cited_text = " ".join(
        source_passages[citation.passage_id]
        for citation in answer.citations
        if citation.passage_id in source_passages
    )
    return token_f1(answer.answer, cited_text)


def citation_correctness(
    answers: list[DraftedAnswer], answer_key: list[AnswerKeyEntry]
) -> float:
    """Return the fraction of cited passage IDs supported by the answer key."""
    expected = {entry.question_id: set(entry.supporting_passage_ids) for entry in answer_key}
    citations = [
        (answer.question_id, citation.passage_id)
        for answer in answers
        for citation in answer.citations
    ]
    if not citations:
        return 0.0
    correct = sum(
        passage_id in expected.get(question_id, set())
        for question_id, passage_id in citations
    )
    return correct / len(citations)


def answer_key_f1(
    answers: list[DraftedAnswer], answer_key: list[AnswerKeyEntry]
) -> float:
    """Return mean token F1 against the hand-authored answer key."""
    expected = {entry.question_id: entry.answer for entry in answer_key}
    scores = [
        token_f1(answer.answer, expected[answer.question_id])
        for answer in answers
        if answer.question_id in expected
    ]
    return sum(scores) / len(scores) if scores else 0.0


def faithfulness_judge(
    answer: DraftedAnswer,
    evidence: str,
    judge_model: str | None = None,
    generator_model: str | None = None,
) -> float:
    """Ask the configured judge model to score evidence faithfulness from 0 to 1."""
    import json

    from .llm import complete, load_config

    config = load_config()
    selected_judge = judge_model or config.get("judge_model")
    if not selected_judge:
        raise ValueError("A judge_model must be configured")
    selected_generator = generator_model or config["models"][0]["name"]
    if selected_judge == selected_generator:
        raise ValueError("The judge model must differ from the generator model")

    response = complete(
        [
            {
                "role": "user",
                "content": (
                    "Score whether the answer is supported by the evidence. "
                    "Return JSON with one numeric field, score, in [0,1].\n"
                    f"Answer: {answer.answer}\nEvidence: {evidence}"
                ),
            }
        ],
        model=selected_judge,
        response_format={"type": "json_object"},
    )
    raw = response["choices"][0]["message"]["content"]
    score = float(json.loads(raw)["score"])
    if not 0 <= score <= 1:
        raise ValueError("Judge score must be between 0 and 1")
    return score


def citation_ids(citations: list[Citation]) -> set[str]:
    """Return cited source passage identifiers."""
    return {citation.passage_id for citation in citations}
