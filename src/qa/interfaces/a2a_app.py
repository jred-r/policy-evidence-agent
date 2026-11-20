"""Small A2A-style HTTP surface for the answer workflow."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..storage.audit import AuditStore
from ..storage.chat import ChatStore
from ..storage.db import connection
from ..storage.review import ReviewStore
from ..workflow.graph import run_workflow

app = FastAPI(title="Policy Evidence Agent", version="0.1.0")
_UI_FILE = Path(__file__).parent / "static" / "index.html"
_app_logger = logging.getLogger("qa")
_app_logger.setLevel(logging.INFO)
_app_logger.propagate = False
if not any(getattr(handler, "_qa_app_handler", False) for handler in _app_logger.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _handler._qa_app_handler = True
    _app_logger.addHandler(_handler)
logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    """Input for one answer task."""

    model_config = ConfigDict(extra="forbid")

    question_id: str | None = None
    question: str


class ChatMessageRequest(BaseModel):
    """Input for one message in a chat session."""

    content: str = Field(min_length=1, max_length=4000)
    live_mode: bool = False


class ReviewDecisionRequest(BaseModel):
    """Human decision for one pending review item."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    edited_answer: str | None = Field(default=None, min_length=1, max_length=4000)


@app.middleware("http")
async def log_http_requests(request, call_next):
    """Log request outcomes without recording bodies, headers, or question text."""
    started = time.monotonic()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    logger.info(
        "http_request method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        path,
        response.status_code,
        (time.monotonic() - started) * 1000,
    )
    return response


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the browser interface."""
    return FileResponse(_UI_FILE)


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
    question_id = (request.question_id or "").strip() or task_id
    try:
        result = run_workflow(
            question_id,
            request.question,
            thread_id=task_id,
        )
    except (ValueError, LookupError) as error:
        raise _workflow_error(error) from error
    return {
        "id": task_id,
        "status": result.get("status", "unknown"),
        "result": result.get("final_answer"),
        "question_id": question_id,
    }


@app.post("/sessions", status_code=201)
def create_session() -> dict[str, Any]:
    """Create a new chat session."""
    return ChatStore().create_session()


@app.get("/sessions")
def list_sessions() -> dict[str, Any]:
    """List saved chat sessions."""
    return {"sessions": ChatStore().list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: UUID) -> dict[str, Any]:
    """Return one session and its messages."""
    session = ChatStore().get_session(str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: UUID) -> Response:
    """Delete a chat and its messages."""
    if not ChatStore().delete_session(str(session_id)):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return Response(status_code=204)


@app.get("/reviews")
def list_reviews() -> dict[str, Any]:
    """List pending human-review items."""
    reviews = ReviewStore().list_pending()
    return {"reviews": reviews, "pending_count": len(reviews)}


@app.post("/reviews/{question_id}/decision")
def decide_review(
    question_id: str,
    request: ReviewDecisionRequest,
) -> dict[str, Any]:
    """Approve or reject a pending item, optionally editing its answer."""
    edited_answer = request.edited_answer
    if edited_answer is not None:
        edited_answer = edited_answer.strip()
        if not edited_answer:
            raise HTTPException(status_code=422, detail="Edited answer cannot be blank")
    try:
        return ReviewStore().decide(
            question_id,
            approved=request.approved,
            edited_answer=edited_answer,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/activity")
def list_activity(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
    """Return recent operational audit events."""
    return {"events": AuditStore().list_recent(limit)}


@app.post("/sessions/{session_id}/messages")
def create_chat_message(
    session_id: UUID,
    request: ChatMessageRequest,
) -> dict[str, Any]:
    """Save a user message, answer it with conversation context, and persist the reply."""
    store = ChatStore()
    selected_session = str(session_id)
    if store.get_session(selected_session) is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    question = request.content.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message cannot be blank")

    history = store.recent_context(selected_session)
    question_id = str(uuid.uuid4())
    store.add_message(
        selected_session,
        "user",
        question,
        {"question_id": question_id, "live_mode": request.live_mode},
    )
    try:
        result = run_workflow(
            question_id,
            question,
            thread_id=str(uuid.uuid4()),
            conversation_history=history,
            llm_mode="record" if request.live_mode else "replay",
        )
    except (ValueError, LookupError) as error:
        event = AuditStore().record(
            "chat.message.failed",
            session_id=selected_session,
            question_id=question_id,
            details={"error_type": type(error).__name__},
        )
        logger.warning(
            "workflow_failed session_id=%s question_id=%s error_type=%s",
            selected_session,
            question_id,
            event["details"]["error_type"],
        )
        raise _workflow_error(error) from error

    status = str(result.get("status", "unknown"))
    answer = result.get("final_answer") or result.get("best_answer")
    if answer:
        content = str(answer["answer"])
        metadata = {
            "status": status,
            "question_id": question_id,
            "confidence": answer.get("confidence"),
            "citations": answer.get("citations", []),
            "live_mode": request.live_mode,
        }
    else:
        content = "I could not produce an evidence-grounded answer."
        metadata = {
            "status": status,
            "question_id": question_id,
            "live_mode": request.live_mode,
        }
    message = store.add_message(selected_session, "assistant", content, metadata)
    return {
        "session_id": selected_session,
        "question_id": question_id,
        "status": status,
        "message": message,
    }


def _workflow_error(error: ValueError | LookupError) -> HTTPException:
    if isinstance(error, LookupError) and str(error).startswith(
        "No recorded model response for key "
    ):
        return HTTPException(
            status_code=503,
            detail=(
                "No saved response is available in replay mode. Set "
                "LLM_MODE=record in the proxy .env and rerun run.ps1 to "
                "generate and save model responses."
            ),
        )
    return HTTPException(status_code=422, detail=str(error))
