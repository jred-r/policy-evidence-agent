from __future__ import annotations

import pytest

from qa.data.retrieve import RetrievedPassage
from qa.domain.schemas import Citation, DraftedAnswer, Question
from qa.workflow import answer as answer_service
from qa.workflow import draft


class FakeRetriever:
    def retrieve(self, question: str, k: int = 5) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                passage_id="P1",
                document_id="policy",
                text="Every privileged request is approved and recorded.",
                distance=0.1,
            )
        ]


def test_off_topic_answer_does_not_enter_policy_review(monkeypatch) -> None:
    captured: list[Question] = []

    def fake_draft_answers_with_sources(questions, retriever, batch_size):
        captured.extend(questions)
        return (
            [
                DraftedAnswer(
                    question_id=questions[0].id,
                    answer="I can help with policy or security questions.",
                    in_scope=False,
                    policy_related=False,
                    citations=[],
                )
            ],
            {questions[0].id: retriever.retrieve(questions[0].text)},
        )

    monkeypatch.setattr(
        answer_service, "draft_answers_with_sources", fake_draft_answers_with_sources
    )
    result = answer_service.answer_question(
        "Q1",
        "What is the weather?",
        retriever=FakeRetriever(),
        conversation_history=[{"role": "user", "content": "Hi"}],
    )

    assert result["status"] == "off_topic"
    assert result["final_answer"]["citations"] == []
    assert result["grounding_f1"] is None
    assert result["human_review_required"] is False
    assert "What is the weather?" in captured[0].text
    assert "Hi" in captured[0].text


def test_prompt_redirects_off_topic_questions_and_grounds_policy_answers(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_call_with_tools(messages, tools, response_model):
        captured["system_message"] = messages[0]["content"]
        return response_model(
            answers=[
                DraftedAnswer(
                    question_id="Q1",
                    answer="I can help with policy or security questions.",
                    in_scope=False,
                    policy_related=False,
                    citations=[],
                )
            ]
        )

    monkeypatch.setattr(draft, "call_with_tools", fake_call_with_tools)
    draft.draft_answers([Question(id="Q1", text="What is the weather?")], FakeRetriever())

    system_message = captured["system_message"]
    assert "friendly, brief redirect" in system_message
    assert "cite the exact supporting passage IDs" in system_message
    assert "policy_related to false" in system_message
    assert "review queue" not in system_message


def test_single_answer_payload_is_normalized_for_one_question(monkeypatch) -> None:
    def fake_call_with_tools(messages, tools, response_model):
        return response_model.model_validate(
            {
                "answer": "Every new account requires manager approval.",
                "in_scope": True,
                "policy_related": True,
                "citations": ["P1"],
            }
        )

    monkeypatch.setattr(draft, "call_with_tools", fake_call_with_tools)
    answers, retrieved = draft.draft_answers_with_sources(
        [Question(id="new-account-controls", text="What controls apply?")],
        FakeRetriever(),
        batch_size=1,
    )

    assert answers[0].question_id == "new-account-controls"
    assert answers[0].citations == [Citation(passage_id="P1")]
    assert retrieved["new-account-controls"][0].passage_id == "P1"


def test_follow_up_context_is_used_for_retrieval_and_drafting(monkeypatch) -> None:
    captured: list[Question] = []

    def fake_draft_answers_with_sources(questions, retriever, batch_size):
        captured.extend(questions)
        return (
            [
                DraftedAnswer(
                    question_id=questions[0].id,
                    answer="Every privileged request is approved and recorded.",
                    in_scope=True,
                    policy_related=True,
                    citations=[Citation(passage_id="P1")],
                )
            ],
            {questions[0].id: retriever.retrieve(questions[0].text)},
        )

    monkeypatch.setattr(
        answer_service, "draft_answers_with_sources", fake_draft_answers_with_sources
    )
    result = answer_service.answer_question(
        "Q2",
        "How often is it reviewed?",
        retriever=FakeRetriever(),
        conversation_history=[
            {"role": "user", "content": "What does the policy say about privileged access?"},
            {"role": "assistant", "content": "Requests are approved and recorded."},
        ],
    )

    assert result["status"] == "answered"
    assert result["grounding_f1"] == 1
    assert result["human_review_required"] is False
    assert "privileged access" in captured[0].text
    assert "How often is it reviewed?" in captured[0].text


@pytest.mark.parametrize(("score", "should_review"), [(0.49, True), (0.5, False)])
def test_grounding_threshold_requires_review_only_below_50_percent(
    monkeypatch, score: float, should_review: bool
) -> None:
    def fake_draft_answers_with_sources(questions, retriever, batch_size):
        return (
            [
                DraftedAnswer(
                    question_id=questions[0].id,
                    answer="Supported policy answer.",
                    in_scope=True,
                    policy_related=True,
                    citations=[Citation(passage_id="P1")],
                )
            ],
            {questions[0].id: retriever.retrieve(questions[0].text)},
        )

    monkeypatch.setattr(
        answer_service, "draft_answers_with_sources", fake_draft_answers_with_sources
    )
    monkeypatch.setattr(answer_service, "grounding_f1", lambda answer, sources: score)

    result = answer_service.answer_question("Q3", "What does policy require?", FakeRetriever())

    assert result["grounding_f1"] == score
    assert result["human_review_required"] is should_review
    if should_review:
        assert result["final_answer"]["answer"] == "This answer is awaiting human review."
        assert result["review_draft"]["answer"] == "Supported policy answer."
    else:
        assert result["final_answer"]["answer"] == "Supported policy answer."
        assert "review_draft" not in result
