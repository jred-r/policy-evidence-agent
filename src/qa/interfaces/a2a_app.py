"""Small A2A-style HTTP surface for policy answers."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..storage.audit import AuditStore
from ..storage.chat import ChatStore
from ..storage.db import connection
from ..storage.review import PENDING_REVIEW_MESSAGE, ReviewStore
from ..workflow.answer import answer_question

app = FastAPI(title="Policy Evidence Agent", version="0.1.0")
_UI_FILE = Path(__file__).parent / "static" / "index.html"
_UI_ASSETS = Path(__file__).parent / "static" / "assets"
app.mount("/assets", StaticFiles(directory=_UI_ASSETS), name="assets")
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
    require_human_review: bool = False


class ReviewDecisionRequest(BaseModel):
    """Human decision for one pending review request."""

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
        result = answer_question(question_id, request.question)
    except (ValueError, LookupError) as error:
        raise _answer_error(error) from error
    status = str(result.get("status", "unknown"))
    review_session_id = None
    if result.get("human_review_required"):
        draft = result.get("review_draft")
        if not draft:
            raise RuntimeError("Answer requires review but its draft is missing")
        review_session_id = ChatStore().create_review_session(
            task_id,
            request.question,
            str(draft["answer"]),
            draft.get("citations", []),
            result.get("grounding_f1"),
            "grounding_threshold",
        )
        status = "pending_review"
    return {
        "id": task_id,
        "status": status,
        "result": result.get("final_answer"),
        "question_id": question_id,
        "review_session_id": review_session_id,
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


@app.get("/activity")
def list_activity(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
    """Return recent operational audit events."""
    return {"events": AuditStore().list_recent(limit)}


@app.get("/review-requests")
def list_review_requests() -> dict[str, Any]:
    """List answers explicitly submitted for human approval."""
    requests = ReviewStore().list_pending()
    return {"requests": requests, "pending_count": len(requests)}


@app.post("/review-requests/{question_id}/decision")
def decide_review_request(
    question_id: str,
    request: ReviewDecisionRequest,
) -> dict[str, Any]:
    """Approve or reject a pending answer, optionally editing its wording."""
    if not request.approved and request.edited_answer is not None:
        raise HTTPException(
            status_code=422,
            detail="An edited answer can only be saved when approving",
        )
    edited_answer = (
        request.edited_answer.strip()
        if request.edited_answer is not None
        else None
    )
    if edited_answer == "":
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
        result = answer_question(
            question_id,
            question,
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
            "chat_answer_failed session_id=%s question_id=%s error_type=%s",
            selected_session,
            question_id,
            event["details"]["error_type"],
        )
        raise _answer_error(error) from error

    status = str(result.get("status", "unknown"))
    answer = result.get("final_answer") or result.get("best_answer")
    if answer:
        grounding_score = result.get("grounding_f1")
        auto_review_required = bool(result.get("human_review_required", False))
        review_required = request.require_human_review or auto_review_required
        review_draft = result.get("review_draft") if auto_review_required else answer
        if review_required and not review_draft:
            raise RuntimeError("Answer requires review but its draft is missing")
        draft_answer = str(review_draft["answer"]) if review_draft else ""
        content = draft_answer
        metadata = {
            "status": status,
            "question_id": question_id,
            "citations": answer.get("citations", []),
            "live_mode": request.live_mode,
        }
        review_request = None
        response_status = status
        if review_required:
            response_status = "pending_review"
            trigger_reason = (
                "grounding_threshold" if auto_review_required else "manual"
            )
            content = PENDING_REVIEW_MESSAGE
            metadata["status"] = response_status
            metadata["human_review_status"] = "pending"
            metadata["citations"] = []
            review_request = {
                "question": question,
                "answer": draft_answer,
                "citations": review_draft.get("citations", []),
                "grounding_f1": grounding_score,
                "trigger_reason": trigger_reason,
            }
    else:
        content = "I could not produce an evidence-grounded answer."
        metadata = {
            "status": status,
            "question_id": question_id,
            "live_mode": request.live_mode,
        }
        review_request = None
        response_status = status
    message = store.add_message(
        selected_session,
        "assistant",
        content,
        metadata,
        review_request=review_request,
    )
    return {
        "session_id": selected_session,
        "question_id": question_id,
        "status": response_status,
        "human_review_status": metadata.get("human_review_status"),
        "message": message,
    }


def _answer_error(error: ValueError | LookupError) -> HTTPException:
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
