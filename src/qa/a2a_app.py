"""Small A2A-style HTTP surface for the answer workflow."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .db import connection
from .graph import run_workflow

app = FastAPI(title="Policy Evidence Agent", version="0.1.0")


class TaskRequest(BaseModel):
    """Input for one answer task."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm the API can reach its PostgreSQL database."""
    with connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "healthy", "database": "connected"}


@app.get("/.well-known/agent.json")
def agent_card() -> dict[str, Any]:
    """Return the public agent card."""
    return {
        "name": "Policy Evidence Agent",
        "description": "Answers security assessment questions with policy citations.",
        "url": "/tasks",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [
            {
                "id": "answer-security-question",
                "name": "Answer security question",
                "description": "Draft a cited answer from policy sources.",
            }
        ],
    }


@app.post("/tasks")
def create_task(request: TaskRequest) -> dict[str, Any]:
    """Execute an answer task and return its result."""
    task_id = str(uuid.uuid4())
    try:
        result = run_workflow(
            request.question_id,
            request.question,
            thread_id=task_id,
        )
    except (ValueError, LookupError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "id": task_id,
        "status": result.get("status", "unknown"),
        "result": result.get("final_answer"),
        "question_id": request.question_id,
    }
