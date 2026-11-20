from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest
from dotenv import load_dotenv

from qa.storage.db import connection


@pytest.fixture(autouse=True)
def isolate_postgres_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    load_dotenv()
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        raise RuntimeError("Set TEST_DATABASE_URL to a disposable PostgreSQL test database")
    if not urlparse(dsn).path.rstrip("/").endswith("_test"):
        raise RuntimeError("TEST_DATABASE_URL must point to a database ending in _test")
    monkeypatch.setenv("DATABASE_URL", dsn)
    with connection(dsn) as conn:
        conn.execute(
            "TRUNCATE qa_llm_responses, qa_llm_daily_budget, qa_review_queue"
        )
