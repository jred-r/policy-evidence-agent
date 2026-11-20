"""PostgreSQL connection and schema management."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS qa_llm_responses (
        cache_key TEXT PRIMARY KEY,
        response JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_llm_daily_budget (
        utc_day DATE PRIMARY KEY,
        request_count INTEGER NOT NULL CHECK (request_count >= 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_audit_events (
        event_id BIGSERIAL PRIMARY KEY,
        event_type TEXT NOT NULL,
        session_id TEXT,
        question_id TEXT,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_policy_passages (
        passage_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        passage_text TEXT NOT NULL,
        embedding JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_chat_sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New chat',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_chat_messages (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES qa_chat_sessions(session_id) ON DELETE CASCADE,
        message_index INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (session_id, message_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_review_requests (
        question_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES qa_chat_sessions(session_id) ON DELETE CASCADE,
        question_text TEXT NOT NULL,
        draft_answer TEXT NOT NULL,
        citations JSONB NOT NULL DEFAULT '[]'::jsonb,
        grounding_f1 DOUBLE PRECISION,
        trigger_reason TEXT NOT NULL DEFAULT 'manual'
            CHECK (trigger_reason IN ('manual', 'grounding_threshold')),
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'approved', 'rejected')),
        decision JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    ALTER TABLE qa_review_requests
        ADD COLUMN IF NOT EXISTS grounding_f1 DOUBLE PRECISION
    """,
    """
    ALTER TABLE qa_review_requests
        ADD COLUMN IF NOT EXISTS trigger_reason TEXT NOT NULL DEFAULT 'manual'
    """,
)


def database_url() -> str:
    """Return the PostgreSQL DSN, loading local environment configuration."""
    load_dotenv()
    value = os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            "DATABASE_URL is required; set it to the PostgreSQL Docker service."
        )
    return value


@contextmanager
def connection(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a PostgreSQL connection and ensure application tables exist."""
    with psycopg.connect(dsn or database_url(), row_factory=dict_row) as conn:
        with conn.transaction():
            for statement in _SCHEMA:
                conn.execute(statement)
        yield conn
