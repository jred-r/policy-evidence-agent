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
    CREATE TABLE IF NOT EXISTS qa_review_queue (
        question_id TEXT PRIMARY KEY,
        payload JSONB NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'approved', 'rejected')),
        decision JSONB,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
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
