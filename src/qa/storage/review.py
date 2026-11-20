"""Persistence for automatic and explicitly requested human review."""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from .audit import emit_event, insert_event
from .db import connection

PENDING_REVIEW_MESSAGE = (
    "This answer is awaiting human approval. It will appear here once a reviewer approves it."
)
REJECTED_REVIEW_MESSAGE = (
    "A human reviewer did not approve this answer. "
    "You can ask a follow-up or contact the policy owner."
)


class ReviewStore:
    """PostgreSQL-backed queue for answers awaiting human approval."""

    def list_pending(self) -> list[dict[str, Any]]:
        """Return answers explicitly submitted for approval."""
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT question_id, session_id, question_text, draft_answer, citations,
                    grounding_f1, trigger_reason, status, created_at
                FROM qa_review_requests
                WHERE status = 'pending'
                ORDER BY created_at, question_id
                """
            ).fetchall()
        return [
            {
                "question_id": row["question_id"],
                "session_id": row["session_id"],
                "question": row["question_text"],
                "answer": row["draft_answer"],
                "citations": row["citations"],
                "grounding_f1": row["grounding_f1"],
                "trigger_reason": row["trigger_reason"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def decide(
        self, question_id: str, approved: bool, edited_answer: str | None = None
    ) -> dict[str, Any]:
        """Approve or reject a pending answer and update its chat message."""
        if edited_answer is not None and not edited_answer.strip():
            raise ValueError("An edited answer cannot be blank")

        with connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT session_id, question_text, draft_answer, citations, status
                    FROM qa_review_requests
                    WHERE question_id = %s
                    FOR UPDATE
                    """,
                    (question_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"No review request for {question_id}")
                if row["status"] != "pending":
                    raise ValueError(f"Review request {question_id} is already decided")

                status = "approved" if approved else "rejected"
                answer = (
                    edited_answer.strip()
                    if approved and edited_answer is not None
                    else row["draft_answer"]
                )
                visible_answer = (
                    answer
                    if approved
                    else REJECTED_REVIEW_MESSAGE
                )
                visible_citations = row["citations"] if approved else []
                conn.execute(
                    """
                    UPDATE qa_review_requests
                    SET draft_answer = %s, status = %s, decision = %s, updated_at = now()
                    WHERE question_id = %s
                    """,
                    (
                        answer,
                        status,
                        Jsonb(
                            {
                                "approved": approved,
                                "edited": approved and edited_answer is not None,
                            }
                        ),
                        question_id,
                    ),
                )
                updated_message = conn.execute(
                    """
                    UPDATE qa_chat_messages
                    SET content = %s,
                        metadata = jsonb_set(
                            jsonb_set(
                                metadata,
                                '{human_review_status}',
                                to_jsonb(%s::text),
                                true
                            ),
                            '{citations}',
                            %s::jsonb,
                            true
                        )
                    WHERE session_id = %s
                        AND role = 'assistant'
                        AND metadata->>'question_id' = %s
                    """,
                    (
                        visible_answer,
                        status,
                        Jsonb(visible_citations),
                        row["session_id"],
                        question_id,
                    ),
                )
                if updated_message.rowcount != 1:
                    raise RuntimeError(
                        f"Review request {question_id} has no corresponding chat answer"
                    )
                event = insert_event(
                    conn,
                    "review.decided",
                    session_id=row["session_id"],
                    question_id=question_id,
                    details={
                        "status": status,
                        "edited": approved and edited_answer is not None,
                    },
                )

        emit_event(event)
        return {
            "question_id": question_id,
            "session_id": row["session_id"],
            "question": row["question_text"],
            "answer": visible_answer,
            "citations": visible_citations,
            "status": status,
        }
