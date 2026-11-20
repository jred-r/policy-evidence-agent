from __future__ import annotations

from pathlib import Path

from qa.data.ingest import validate_data
from qa.data.retrieve import PolicyRetriever, RetrievedPassage
from qa.domain.schemas import Citation, DraftedAnswer
from qa.workflow.scoring import grounding_f1, token_f1


class FakeCollection:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict[str, str]]] = []

    def upsert(self, **kwargs: object) -> None:
        ids = kwargs["ids"]
        documents = kwargs["documents"]
        metadatas = kwargs["metadatas"]
        self.rows = [
            (str(ids[index]), str(documents[index]), metadatas[index])
            for index in range(len(ids))
        ]

    def query(self, **kwargs: object) -> dict[str, object]:
        selected = self.rows[: int(kwargs["n_results"])]
        return {
            "ids": [[row[0] for row in selected]],
            "documents": [[row[1] for row in selected]],
            "metadatas": [[row[2] for row in selected]],
            "distances": [[0.1 for _row in selected]],
        }


def test_synthetic_dataset_has_40_valid_question_answer_pairs() -> None:
    dataset = validate_data(Path("data"))

    assert len(dataset.questions) == 40
    assert len(dataset.answer_key) == 40
    assert {entry.question_id for entry in dataset.answer_key} == {
        question.id for question in dataset.questions
    }


def test_local_retrieval_returns_chroma_shaped_citations() -> None:
    dataset = validate_data(Path("data"))
    collection = FakeCollection()
    retriever = PolicyRetriever(
        collection=collection,
        embedder=lambda texts: [[float(len(text))] for text in texts],
    )
    retriever.ingest(dataset.policies)

    found = retriever.retrieve(dataset.questions[0].text, k=1)

    assert found == [
        RetrievedPassage(
            passage_id="P001",
            document_id="synthetic-security-policy",
            text=dataset.policies[0].passages[0].text,
            distance=0.1,
        )
    ]


def test_grounding_f1_uses_only_cited_passages() -> None:
    answer = DraftedAnswer(
        question_id="Q1",
        answer="Admin access requires MFA.",
        confidence=0.9,
        citations=[Citation(passage_id="P2")],
    )

    assert grounding_f1(answer, {"P2": "Admin access requires MFA."}) == 1.0
    assert grounding_f1(answer, {"P1": "Unrelated policy wording."}) == 0.0
    assert token_f1("MFA is required", "MFA is required") == 1.0
