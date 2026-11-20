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
        in_scope=True,
        policy_related=True,
        citations=[Citation(passage_id=passage_id)],
    )


def test_evaluation_reports_single_pass_answers_and_off_topic_results(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = load_data(Path("data"))
    answers = [
        _draft(
            question.id,
            next(
                item.answer
                for item in dataset.answer_key
                if item.question_id == question.id
            ),
            f"P{int(question.id[1:]):03}",
        )
        for question in dataset.questions
    ]
    answers[0] = DraftedAnswer(
        question_id=answers[0].question_id,
        answer="I can help with policy questions.",
        in_scope=False,
        policy_related=False,
        citations=[],
    )
    calls = []

    def draft_answers(questions, retriever, batch_size):
        calls.append(len(questions))
        return answers

    monkeypatch.setattr(evaluation, "draft_answers", draft_answers)

    result = evaluation.evaluate(
        dataset=dataset,
        retriever=FixtureRetriever(dataset),
        output_dir=tmp_path / "results",
    )

    assert calls == [len(dataset.questions)]
    assert result["out_of_scope_percent"] > 0
    assert result["answers"][0]["status"] == "off_topic"
    assert (tmp_path / "results" / "evaluation.json").exists()
    assert (tmp_path / "results" / "summary.txt").exists()
