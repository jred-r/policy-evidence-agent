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
        "/tasks", json={"question_id": "Q1", "question": "Can you cite evidence?"}
    )

    assert home.status_code == 200
    assert 'id="question-form"' in home.text
    assert card.status_code == 200
    assert card.json()["name"] == "Policy Evidence Agent"
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["database"] == "connected"
    assert task.status_code == 200
    assert task.json()["status"] == "answered"
    assert task.json()["result"]["answer"] == "Can you cite evidence?"


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
