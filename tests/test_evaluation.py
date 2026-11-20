from __future__ import annotations

from pathlib import Path

from qa.data.retrieve import RetrievedPassage
from qa.domain.schemas import Citation, DataSet, DraftedAnswer, load_data
from qa.evaluation import runner as evaluation


class FixtureRetriever:
    def __init__(self, dataset: DataSet) -> None:
        self._by_text = {
            question.text: question.id for question in dataset.questions
        }
        self._passages = {
            passage.id: passage
            for policy in dataset.policies
            for passage in policy.passages
        }

    def ingest(self, documents: object) -> int:
        return len(self._passages)

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedPassage]:
        question_id = self._by_text[question]
        passage_id = f"P{int(question_id[1:]):03}"
        passage = self._passages[passage_id]
        return [
            RetrievedPassage(
                passage_id=passage.id,
                document_id=passage.document_id,
                text=passage.text,
                distance=0.0,
            )
        ]


def _draft(question_id: str, answer: str, passage_id: str) -> DraftedAnswer:
    return DraftedAnswer(
        question_id=question_id,
        answer=answer,
        confidence=0.8,
        citations=[Citation(passage_id=passage_id)],
    )


def test_evaluation_reports_retries_and_escalations(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = load_data(Path("data"))
    initial = [
        _draft(
            question.id,
            "Unsupported draft." if question.id in {"Q001", "Q002"} else
            next(item.answer for item in dataset.answer_key if item.question_id == question.id),
            f"P{int(question.id[1:]):03}",
        )
        for question in dataset.questions
    ]
    retried = [
        _draft("Q001", dataset.answer_key[0].answer, "P001"),
        _draft("Q002", "Still unsupported.", "P002"),
    ]
    calls = 0

    def draft_answers(questions, retriever, batch_size):
        nonlocal calls
        calls += 1
        return initial if calls == 1 else retried

    monkeypatch.setattr(evaluation, "draft_answers", draft_answers)

    result = evaluation.evaluate(
        dataset=dataset,
        retriever=FixtureRetriever(dataset),
        output_dir=tmp_path / "results",
    )

    assert result["first_attempt_pass_rate"] == 0.95
    assert result["f1_after_retries"] == 0.975
    assert result["escalated_percent"] == 2.5
    assert result["answers"][0]["attempts"] == 2
    assert result["answers"][1]["status"] == "review"
    assert (tmp_path / "results" / "evaluation.json").exists()
    assert (tmp_path / "results" / "summary.txt").exists()
