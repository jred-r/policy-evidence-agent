"""Persistent audit events and application log output."""

from __future__ import annotations

import json
import logging
from typing import Any

from psycopg.types.json import Jsonb

from .db import connection

logger = logging.getLogger("qa.audit")


def insert_event(
    conn: Any,
    event_type: str,
    *,
    session_id: str | None = None,
    question_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert an audit event in the caller's transaction."""
    row = conn.execute(
        """
        INSERT INTO qa_audit_events
            (event_type, session_id, question_id, details)
        VALUES (%s, %s, %s, %s)
        RETURNING event_id, event_type, session_id, question_id, details, created_at
        """,
        (event_type, session_id, question_id, Jsonb(details or {})),
    ).fetchone()
    return dict(row)


def emit_event(event: dict[str, Any]) -> None:
    """Write a compact JSON event to the container's standard log stream."""
    logger.info("audit_event %s", json.dumps(event, default=str, separators=(",", ":")))


class AuditStore:
    """Read recent persisted audit events."""

    def record(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        question_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with connection() as conn:
            event = insert_event(
                conn,
                event_type,
                session_id=session_id,
                question_id=question_id,
                details=details,
            )
        emit_event(event)
        return event

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, session_id, question_id, details, created_at
                FROM qa_audit_events
                ORDER BY event_id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
