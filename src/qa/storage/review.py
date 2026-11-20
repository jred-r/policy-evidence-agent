"""Persistent queue for human answer review."""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from .audit import emit_event, insert_event
from .db import connection


class ReviewStore:
    """PostgreSQL-backed review queue and decision log."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn

    def enqueue(self, question_id: str, payload: dict[str, Any]) -> None:
        """Insert or replace a pending answer in the review queue."""
        answer = payload.get("answer")
        confidence = answer.get("confidence") if isinstance(answer, dict) else None
        with connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO qa_review_queue
                        (question_id, payload, status, decision, updated_at)
                    VALUES (%s, %s, 'pending', NULL, now())
                    ON CONFLICT (question_id) DO UPDATE SET
                        payload = EXCLUDED.payload,
                        status = 'pending',
                        decision = NULL,
                        updated_at = now()
                    """,
                    (question_id, Jsonb(payload)),
                )
                audit_event = insert_event(
                    conn,
                    "review.queued",
                    question_id=question_id,
                    details={"confidence": confidence},
                )
        emit_event(audit_event)

    def list_pending(self) -> list[dict[str, Any]]:
        """Return pending review items in stable question order."""
        with connection(self.dsn) as conn:
            rows = conn.execute(
                "SELECT question_id, payload, status, updated_at FROM qa_review_queue "
                "WHERE status='pending' ORDER BY updated_at DESC, question_id"
            ).fetchall()
        return [
            {
                "question_id": row["question_id"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                **row["payload"],
            }
            for row in rows
        ]

    def decide(
        self, question_id: str, approved: bool, edited_answer: str | None = None
    ) -> dict[str, Any]:
        """Persist approval or an edited answer and return the final item."""
        with connection(self.dsn) as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT payload, status FROM qa_review_queue "
                    "WHERE question_id=%s FOR UPDATE",
                    (question_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"No review item for {question_id}")
                if row["status"] != "pending":
                    raise ValueError(f"Review item {question_id} is already decided")
                payload = row["payload"]
                if edited_answer is not None:
                    payload["answer"] = edited_answer
                status = "approved" if approved else "rejected"
                conn.execute(
                    """
                    UPDATE qa_review_queue
                    SET payload=%s, status=%s, decision=%s, updated_at=now()
                    WHERE question_id=%s
                    """,
                    (
                        Jsonb(payload),
                        status,
                        Jsonb(
                            {
                                "approved": approved,
                                "edited": edited_answer is not None,
                            }
                        ),
                        question_id,
                    ),
                )
                if edited_answer is not None:
                    conn.execute(
                        """
                        UPDATE qa_chat_messages
                        SET content=%s,
                            metadata=jsonb_set(metadata, '{status}', to_jsonb(%s::text), true)
                        WHERE metadata->>'question_id'=%s
                        """,
                        (edited_answer, status, question_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE qa_chat_messages
                        SET metadata=jsonb_set(metadata, '{status}', to_jsonb(%s::text), true)
                        WHERE metadata->>'question_id'=%s
                        """,
                        (status, question_id),
                    )
                audit_event = insert_event(
                    conn,
                    "review.decided",
                    question_id=question_id,
                    details={
                        "status": status,
                        "edited": edited_answer is not None,
                    },
                )
        emit_event(audit_event)
        return {"question_id": question_id, **payload, "status": status}

    def get_final(self, question_id: str) -> dict[str, Any]:
        """Return an approved item for export."""
        with connection(self.dsn) as conn:
            row = conn.execute(
                "SELECT payload, status FROM qa_review_queue WHERE question_id=%s",
                (question_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No review item for {question_id}")
        if row["status"] != "approved":
            raise ValueError(f"Review item {question_id} is not approved")
        return {"question_id": question_id, **row["payload"]}
