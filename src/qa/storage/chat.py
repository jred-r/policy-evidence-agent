"""PostgreSQL-backed chat sessions and message history."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from psycopg.types.json import Jsonb

from .audit import emit_event, insert_event
from .db import connection

Role = Literal["user", "assistant"]


class ChatStore:
    """Persist conversations and their messages in PostgreSQL."""

    def create_session(self) -> dict[str, Any]:
        """Create an empty chat session."""
        session_id = str(uuid.uuid4())
        with connection() as conn:
            row = conn.execute(
                """
                INSERT INTO qa_chat_sessions (session_id)
                VALUES (%s)
                RETURNING title, created_at, updated_at
                """,
                (session_id,),
            ).fetchone()
            audit_event = insert_event(conn, "chat.created", session_id=session_id)
        emit_event(audit_event)
        return {
            "session_id": session_id,
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": [],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return sessions ordered by most recent activity."""
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                    count(m.message_id) AS message_count
                FROM qa_chat_sessions AS s
                LEFT JOIN qa_chat_messages AS m USING (session_id)
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC, s.session_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a session and its messages, or None if it does not exist."""
        with connection() as conn:
            session = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM qa_chat_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            messages = conn.execute(
                """
                SELECT m.message_id, m.role, m.content,
                    CASE
                        WHEN m.metadata->>'status' = 'review'
                            AND r.status IN ('approved', 'rejected')
                        THEN jsonb_set(m.metadata, '{status}', to_jsonb(r.status), true)
                        ELSE m.metadata
                    END AS metadata,
                    m.created_at
                FROM qa_chat_messages AS m
                LEFT JOIN qa_review_queue AS r
                    ON r.question_id = m.metadata->>'question_id'
                WHERE m.session_id = %s
                ORDER BY m.message_index
                """,
                (session_id,),
            ).fetchall()
        return {**session, "messages": [dict(message) for message in messages]}

    def recent_context(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        """Return recent turns in chronological order for follow-up questions."""
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM qa_chat_messages
                WHERE session_id = %s
                ORDER BY message_index DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def add_message(
        self,
        session_id: str,
        role: Role,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a user or assistant message and update session activity."""
        message_id = str(uuid.uuid4())
        audit_event = None
        with connection() as conn:
            with conn.transaction():
                session = conn.execute(
                    """
                    SELECT title FROM qa_chat_sessions
                    WHERE session_id = %s
                    FOR UPDATE
                    """,
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"No chat session {session_id}")
                message_index = conn.execute(
                    """
                    SELECT COALESCE(MAX(message_index), 0) + 1 AS next_index
                    FROM qa_chat_messages
                    WHERE session_id = %s
                    """,
                    (session_id,),
                ).fetchone()["next_index"]
                row = conn.execute(
                    """
                    INSERT INTO qa_chat_messages
                        (message_id, session_id, message_index, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING created_at
                    """,
                    (
                        message_id,
                        session_id,
                        message_index,
                        role,
                        content,
                        Jsonb(metadata or {}),
                    ),
                ).fetchone()
                title = session["title"]
                if role == "user" and title == "New chat":
                    normalized = " ".join(content.split())
                    title = normalized[:57] + "..." if len(normalized) > 60 else normalized
                conn.execute(
                    """
                    UPDATE qa_chat_sessions
                    SET title = %s, updated_at = now()
                    WHERE session_id = %s
                    """,
                    (title, session_id),
                )
                if role == "user" and metadata and metadata.get("question_id"):
                    audit_event = insert_event(
                        conn,
                        "chat.message.received",
                        session_id=session_id,
                        question_id=str(metadata["question_id"]),
                        details={"live_mode": bool(metadata.get("live_mode", False))},
                    )
                elif role == "assistant" and metadata and metadata.get("question_id"):
                    audit_event = insert_event(
                        conn,
                        "chat.message.completed",
                        session_id=session_id,
                        question_id=str(metadata["question_id"]),
                        details={
                            "status": metadata.get("status", "unknown"),
                            "live_mode": bool(metadata.get("live_mode", False)),
                            "confidence": metadata.get("confidence"),
                        },
                    )
        if audit_event:
            emit_event(audit_event)
        return {
            "message_id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": row["created_at"],
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages while preserving a deletion event."""
        with connection() as conn:
            with conn.transaction():
                deleted = conn.execute(
                    """
                    DELETE FROM qa_chat_sessions
                    WHERE session_id = %s
                    RETURNING session_id
                    """,
                    (session_id,),
                ).fetchone()
                if deleted is None:
                    return False
                audit_event = insert_event(
                    conn,
                    "chat.deleted",
                    session_id=session_id,
                )
        emit_event(audit_event)
        return True
