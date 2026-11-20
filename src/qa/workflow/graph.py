"""Checkpointed answer workflow with grounding retries and review escalation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ..data.retrieve import PolicyRetriever, RetrievedPassage
from ..domain.schemas import DraftedAnswer, Question
from ..llm.client import override_llm_mode
from ..storage.db import database_url
from ..storage.review import ReviewStore
from .draft import contextualize_question, draft_answers
from .scoring import grounding_f1


class WorkflowState(TypedDict, total=False):
    """Persisted workflow state."""

    question_id: str
    question: str
    passages: list[dict[str, Any]]
    answer: dict[str, Any]
    best_answer: dict[str, Any]
    score: float
    best_score: float
    attempts: int
    status: str
    final_answer: dict[str, Any]


DraftFunction = Callable[[Question, list[RetrievedPassage]], DraftedAnswer]


def create_workflow(
    retriever: PolicyRetriever | None = None,
    review_store: ReviewStore | None = None,
    checkpointer: Any = None,
    draft_function: DraftFunction | None = None,
    interrupt_before: list[str] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    llm_mode: Literal["record", "replay"] | None = None,
) -> Any:
    """Build the resumable parse/retrieve/draft/score/retry/review graph."""
    policy_retriever = retriever or PolicyRetriever()
    reviews = review_store or ReviewStore()
    history = conversation_history or []

    def parse(state: WorkflowState) -> dict[str, Any]:
        if not state.get("question_id") or not state.get("question"):
            raise ValueError("question_id and question are required")
        return {"attempts": int(state.get("attempts", 0))}

    def retrieve(state: WorkflowState) -> dict[str, Any]:
        question = contextualize_question(state["question"], history)
        passages = policy_retriever.retrieve(question, k=5)
        return {
            "passages": [
                {
                    "passage_id": item.passage_id,
                    "document_id": item.document_id,
                    "text": item.text,
                    "distance": item.distance,
                }
                for item in passages
            ]
        }

    def draft(state: WorkflowState) -> dict[str, Any]:
        question = Question(
            id=state["question_id"],
            text=contextualize_question(state["question"], history),
        )
        passages = [
            RetrievedPassage(**item) for item in state.get("passages", [])
        ]
        if draft_function:
            answer = draft_function(question, passages)
        elif llm_mode is None:
            answer = draft_answers([question], policy_retriever, batch_size=1)[0]
        else:
            with override_llm_mode(llm_mode):
                answer = draft_answers([question], policy_retriever, batch_size=1)[0]
        if answer.question_id != question.id:
            raise ValueError("Draft returned an answer for a different question")
        return {"answer": answer.model_dump(), "attempts": state.get("attempts", 0) + 1}

    def score(state: WorkflowState) -> dict[str, Any]:
        answer = DraftedAnswer.model_validate(state["answer"])
        source_text = {item["passage_id"]: item["text"] for item in state["passages"]}
        current_score = grounding_f1(answer, source_text)
        previous_score = float(state.get("best_score", -1.0))
        if current_score >= previous_score:
            best_answer = answer.model_dump()
            best_score = current_score
        else:
            best_answer = state["best_answer"]
            best_score = previous_score

        if current_score >= 0.95:
            return {
                "score": current_score,
                "best_score": best_score,
                "best_answer": best_answer,
                "final_answer": best_answer,
                "status": "answered",
            }
        if state["attempts"] < 2:
            status = "retry"
        elif best_score > 0.90:
            status = "answered"
        else:
            status = "review"
        updates: dict[str, Any] = {
            "score": current_score,
            "best_score": best_score,
            "best_answer": best_answer,
            "status": status,
        }
        if status == "answered":
            updates["final_answer"] = best_answer
        return updates

    def review(state: WorkflowState) -> dict[str, Any]:
        item = state.get("best_answer") or state["answer"]
        reviews.enqueue(
            state["question_id"],
            {
                "question": state["question"],
                "conversation_history": history,
                "answer": item,
                "grounding_f1": state.get("best_score", state.get("score", 0.0)),
                "passages": state.get("passages", []),
            },
        )
        return {"status": "review"}

    builder = StateGraph(WorkflowState)
    builder.add_node("parse", parse)
    builder.add_node("retrieve", retrieve)
    builder.add_node("draft", draft)
    builder.add_node("score", score)
    builder.add_node("review", review)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "score")
    builder.add_conditional_edges(
        "score",
        lambda state: state["status"],
        {"retry": "draft", "review": "review", "answered": END},
    )
    builder.add_edge("review", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )


def run_workflow(
    question_id: str,
    question: str,
    thread_id: str,
    retriever: PolicyRetriever | None = None,
    review_store: ReviewStore | None = None,
    draft_function: DraftFunction | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    llm_mode: Literal["record", "replay"] | None = None,
) -> WorkflowState:
    """Run or resume a workflow using a persistent PostgreSQL checkpoint."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        graph = create_workflow(
            retriever=retriever,
            review_store=review_store,
            checkpointer=saver,
            draft_function=draft_function,
            conversation_history=conversation_history,
            llm_mode=llm_mode,
        )
        return graph.invoke(
            {"question_id": question_id, "question": question},
            config={"configurable": {"thread_id": thread_id}},
        )
