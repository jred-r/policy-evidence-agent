from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from qa.interfaces import a2a_app, mcp_server
from qa.storage.review import ReviewStore


def test_review_decision_persists_edited_final_answer() -> None:
    store = ReviewStore()
    store.enqueue("Q1", {"answer": "Draft", "confidence": 0.4})

    result = store.decide("Q1", approved=True, edited_answer="Human-approved answer")

    assert result["answer"] == "Human-approved answer"
    assert store.list_pending() == []
    assert store.get_final("Q1")["answer"] == "Human-approved answer"


def test_agent_card_and_task_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "run_workflow",
        lambda question_id, question, thread_id: {
            "status": "answered",
            "final_answer": {"question_id": question_id, "answer": question},
        },
    )
    client = TestClient(a2a_app.app)

    home = client.get("/")
    card = client.get("/.well-known/agent.json")
    task = client.post(
        "/tasks", json={"question": "Can you cite evidence?"}
    )
    task_with_id = client.post(
        "/tasks", json={"question_id": "Q1", "question": "Can you cite evidence?"}
    )

    assert home.status_code == 200
    assert 'id="composer"' in home.text
    assert 'id="session-list"' in home.text
    assert 'id="question-id"' not in home.text
    assert card.status_code == 200
    assert card.json()["name"] == "Policy Evidence Agent"
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["database"] == "connected"
    assert task.status_code == 200
    assert task.json()["status"] == "answered"
    assert task.json()["result"]["answer"] == "Can you cite evidence?"
    assert task.json()["question_id"] == task.json()["id"]
    assert task.json()["result"]["question_id"] == task.json()["question_id"]
    assert task_with_id.status_code == 200
    assert task_with_id.json()["question_id"] == "Q1"


def test_chat_sessions_save_history_and_support_follow_up(monkeypatch) -> None:
    calls = []

    def fake_workflow(
        question_id,
        question,
        thread_id,
        conversation_history=None,
        llm_mode=None,
    ):
        calls.append((question, conversation_history, llm_mode))
        return {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": f"Answer: {question}",
                "confidence": 0.9,
                "citations": [{"passage_id": "P1", "quote": "Policy evidence"}],
            },
        }

    monkeypatch.setattr(a2a_app, "run_workflow", fake_workflow)
    client = TestClient(a2a_app.app)

    session = client.post("/sessions")
    session_id = session.json()["session_id"]
    first = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "Where is access control defined?"},
    )
    second = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "How often is it reviewed?", "live_mode": True},
    )
    saved = client.get(f"/sessions/{session_id}").json()
    sessions = client.get("/sessions").json()["sessions"]
    new_session = client.post("/sessions").json()

    assert session.status_code == 201
    assert first.status_code == second.status_code == 200
    assert second.json()["message"]["metadata"]["citations"][0]["passage_id"] == "P1"
    assert [message["role"] for message in saved["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert calls[0][1] == []
    assert calls[1][1] == [
        {"role": "user", "content": "Where is access control defined?"},
        {"role": "assistant", "content": "Answer: Where is access control defined?"},
    ]
    assert calls[0][2] == "replay"
    assert calls[1][2] == "record"
    assert sessions[0]["title"] == "Where is access control defined?"
    assert new_session["messages"] == []


def test_chat_replay_miss_returns_actionable_message(monkeypatch) -> None:
    cache_key = "a" * 64

    def missing_response(*args, **kwargs):
        raise LookupError(f"No recorded model response for key {cache_key}")

    monkeypatch.setattr(a2a_app, "run_workflow", missing_response)
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What does the policy say?"},
    )

    assert response.status_code == 503
    assert cache_key not in response.text
    assert "LLM_MODE=record" in response.json()["detail"]


def test_mcp_registers_and_calls_all_required_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "run_workflow",
        lambda question_id, question, thread_id: {
            "status": "answered",
            "final_answer": {"question_id": question_id, "answer": question},
        },
    )

    class FakeReviewStore:
        def list_pending(self):
            return [{"question_id": "Q1"}]

        def decide(self, question_id, approved, edited_answer=None):
            return {
                "question_id": question_id,
                "approved": approved,
                "answer": edited_answer,
            }

    monkeypatch.setattr(mcp_server, "ReviewStore", FakeReviewStore)

    async def invoke_tools():
        names = {tool.name for tool in await mcp_server.mcp.list_tools()}
        assert names == {"answer_question", "list_review_queue", "approve_answer"}
        answers = await mcp_server.mcp.call_tool(
            "answer_question", {"question_id": "Q1", "question": "Can you cite evidence?"}
        )
        queue = await mcp_server.mcp.call_tool("list_review_queue", {})
        approved = await mcp_server.mcp.call_tool(
            "approve_answer",
            {"question_id": "Q1", "approved": True, "edited_answer": "Approved"},
        )
        assert answers
        assert queue
        assert approved

    asyncio.run(invoke_tools())
