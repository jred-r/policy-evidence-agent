from __future__ import annotations

import os
import uuid

from langgraph.checkpoint.postgres import PostgresSaver

from qa.data.retrieve import RetrievedPassage
from qa.domain.schemas import Citation, DraftedAnswer, Question
from qa.storage.review import ReviewStore
from qa.workflow.graph import create_workflow


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


def _answer(question: Question, text: str) -> DraftedAnswer:
    return DraftedAnswer(
        question_id=question.id,
        answer=text,
        confidence=0.8,
        citations=[Citation(passage_id="P1")],
    )


def test_checkpoint_resume_after_process_restart() -> None:
    dsn = os.environ["DATABASE_URL"]
    thread_id = str(uuid.uuid4())
    retriever = FakeRetriever()

    def draft(question: Question, passages: list[RetrievedPassage]) -> DraftedAnswer:
        return _answer(question, "Every privileged request is approved and recorded.")

    with PostgresSaver.from_conn_string(dsn) as saver:
        saver.setup()
        graph = create_workflow(
            retriever=retriever,
            review_store=ReviewStore(),
            checkpointer=saver,
            draft_function=draft,
            interrupt_before=["score"],
        )
        interrupted = graph.invoke(
            {"question_id": "Q1", "question": "Are privileged requests recorded?"},
            config={"configurable": {"thread_id": thread_id}},
        )
        assert interrupted["answer"]["question_id"] == "Q1"
        assert "final_answer" not in interrupted

    with PostgresSaver.from_conn_string(dsn) as saver:
        saver.setup()
        resumed_graph = create_workflow(
            retriever=retriever,
            review_store=ReviewStore(),
            checkpointer=saver,
            draft_function=draft,
        )
        resumed = resumed_graph.invoke(
            None, config={"configurable": {"thread_id": thread_id}}
        )

    assert resumed["status"] == "answered"
    assert resumed["final_answer"]["answer"] == interrupted["answer"]["answer"]


def test_low_grounding_after_retry_enters_persistent_review_queue() -> None:
    thread_id = str(uuid.uuid4())
    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        saver.setup()
        store = ReviewStore()
        graph = create_workflow(
            retriever=FakeRetriever(),
            review_store=store,
            checkpointer=saver,
            draft_function=lambda question, passages: _answer(question, "Unsupported."),
        )
        result = graph.invoke(
            {"question_id": "Q2", "question": "Are privileged requests recorded?"},
            config={"configurable": {"thread_id": thread_id}},
        )

    assert result["status"] == "review"
    assert len(store.list_pending()) == 1
    assert store.list_pending()[0]["question_id"] == "Q2"
