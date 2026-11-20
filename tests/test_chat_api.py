from __future__ import annotations

import asyncio
import re

from fastapi.testclient import TestClient

from qa.interfaces import a2a_app, mcp_server
from qa.storage.audit import AuditStore
from qa.storage.chat import ChatStore


def test_delete_chat_session_removes_messages_and_logs_event() -> None:
    store = ChatStore()
    session_id = store.create_session()["session_id"]
    store.add_message(session_id, "user", "Remove this")
    client = TestClient(a2a_app.app)

    deleted = client.delete(f"/sessions/{session_id}")

    assert deleted.status_code == 204
    assert client.get(f"/sessions/{session_id}").status_code == 404
    events = client.get("/activity").json()["events"]
    assert any(
        event["event_type"] == "chat.deleted" and event["session_id"] == session_id
        for event in events
    )
    assert client.delete(f"/sessions/{session_id}").status_code == 404


def test_agent_card_and_task_endpoint_use_normal_chat_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question: {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": question,
                "in_scope": True,
                "citations": [],
            },
        },
    )
    client = TestClient(a2a_app.app)

    home = client.get("/")
    card = client.get("/.well-known/agent.json")
    task = client.post("/tasks", json={"question": "Can you cite evidence?"})

    assert home.status_code == 200
    assert '<div id="root"></div>' in home.text
    assert "/assets/" in home.text
    for asset_path in re.findall(r'(?:src|href)="(/assets/[^"]+)"', home.text):
        assert client.get(asset_path).status_code == 200
    assert card.status_code == 200
    assert card.json()["name"] == "Policy Evidence Agent"
    assert client.get("/health").json()["database"] == "connected"
    assert task.status_code == 200
    assert task.json()["status"] == "answered"
    assert task.json()["result"]["answer"] == "Can you cite evidence?"
    assert task.json()["question_id"] == task.json()["id"]
    assert client.get("/review-requests").json()["pending_count"] == 0


def test_chat_sessions_save_history_and_support_follow_up(monkeypatch) -> None:
    calls = []

    def fake_answer(
        question_id,
        question,
        conversation_history=None,
        llm_mode=None,
    ):
        calls.append((question, conversation_history, llm_mode))
        return {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": f"Answer: {question}",
                "in_scope": True,
                "citations": [{"passage_id": "P1", "quote": "Policy evidence"}],
            },
        }

    monkeypatch.setattr(a2a_app, "answer_question", fake_answer)
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
    assert second.json()["human_review_status"] is None
    assert sessions[0]["title"] == "Where is access control defined?"
    assert new_session["messages"] == []


def test_opted_in_answer_can_be_edited_and_approved(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question, **kwargs: {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": "Draft response.",
                "in_scope": True,
                "citations": [{"passage_id": "P1", "quote": "Policy evidence"}],
            },
        },
    )
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]

    sent = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What does the policy require?", "require_human_review": True},
    )
    pending = client.get("/review-requests").json()
    question_id = sent.json()["question_id"]
    decision = client.post(
        f"/review-requests/{question_id}/decision",
        json={"approved": True, "edited_answer": "Approved response."},
    )
    saved = client.get(f"/sessions/{session_id}").json()
    activity = client.get("/activity").json()["events"]

    assert sent.status_code == 200
    assert sent.json()["human_review_status"] == "pending"
    assert sent.json()["message"]["metadata"]["human_review_status"] == "pending"
    assert pending["pending_count"] == 1
    assert pending["requests"][0]["question"] == "What does the policy require?"
    assert pending["requests"][0]["answer"] == "Draft response."
    assert sent.json()["message"]["content"] != "Draft response."
    assert sent.json()["message"]["metadata"]["citations"] == []
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
    assert saved["messages"][-1]["content"] == "Approved response."
    assert saved["messages"][-1]["metadata"]["human_review_status"] == "approved"
    assert client.get("/review-requests").json()["pending_count"] == 0
    assert {"review.requested", "review.decided"} <= {
        event["event_type"] for event in activity
    }


def test_low_grounding_answer_is_hidden_until_review_is_complete(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question, **kwargs: {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": "This answer is awaiting human review.",
                "in_scope": True,
                "policy_related": True,
                "citations": [],
            },
            "grounding_f1": 0.42,
            "human_review_required": True,
            "review_draft": {
                "question_id": question_id,
                "answer": "Unreviewed private draft.",
                "in_scope": True,
                "policy_related": True,
                "citations": [{"passage_id": "P1", "quote": "Policy evidence"}],
            },
        },
    )
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]

    sent = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What does this policy require?"},
    )
    pending = client.get("/review-requests").json()["requests"][0]
    saved_pending = client.get(f"/sessions/{session_id}").json()["messages"][-1]

    assert sent.status_code == 200
    assert sent.json()["status"] == "pending_review"
    assert sent.json()["message"]["content"] != "Unreviewed private draft."
    assert saved_pending["content"] != "Unreviewed private draft."
    assert saved_pending["metadata"]["citations"] == []
    assert pending["answer"] == "Unreviewed private draft."
    assert pending["grounding_f1"] == 0.42
    assert pending["trigger_reason"] == "grounding_threshold"

    approved = client.post(
        f"/review-requests/{sent.json()['question_id']}/decision",
        json={"approved": True},
    )
    saved_approved = client.get(f"/sessions/{session_id}").json()["messages"][-1]

    assert approved.status_code == 200
    assert saved_approved["content"] == "Unreviewed private draft."
    assert saved_approved["metadata"]["citations"][0]["passage_id"] == "P1"
    assert saved_approved["metadata"]["human_review_status"] == "approved"


def test_human_reviewer_can_reject_a_pending_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question, **kwargs: {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": "Draft response.",
                "in_scope": True,
                "citations": [],
            },
        },
    )
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]
    sent = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "Review this answer", "require_human_review": True},
    )

    rejected = client.post(
        f"/review-requests/{sent.json()['question_id']}/decision",
        json={"approved": False},
    )
    saved = client.get(f"/sessions/{session_id}").json()

    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert saved["messages"][-1]["metadata"]["human_review_status"] == "rejected"
    assert "did not approve" in saved["messages"][-1]["content"]
    assert saved["messages"][-1]["content"] != "Draft response."


def test_task_auto_review_response_does_not_expose_draft(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question: {
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": "This answer is awaiting human review.",
                "in_scope": True,
                "policy_related": True,
                "citations": [],
            },
            "grounding_f1": 0.2,
            "human_review_required": True,
            "review_draft": {
                "answer": "A task draft that must stay hidden.",
                "citations": [],
            },
        },
    )
    client = TestClient(a2a_app.app)

    task = client.post("/tasks", json={"question": "Can you cite evidence?"})

    assert task.status_code == 200
    assert task.json()["status"] == "pending_review"
    assert "A task draft that must stay hidden." not in task.text
    assert task.json()["review_session_id"]
    assert client.get("/review-requests").json()["pending_count"] == 1


def test_off_topic_reply_is_saved_without_citations(monkeypatch) -> None:
    monkeypatch.setattr(
        a2a_app,
        "answer_question",
        lambda question_id, question, **kwargs: {
            "status": "off_topic",
            "final_answer": {
                "question_id": question_id,
                "answer": "I can help with policy or security questions.",
                "in_scope": False,
                "citations": [],
            },
        },
    )
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What is the weather?"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "off_topic"
    assert response.json()["message"]["metadata"]["citations"] == []


def test_chat_replay_miss_returns_actionable_message(monkeypatch) -> None:
    cache_key = "a" * 64

    def missing_response(*args, **kwargs):
        raise LookupError(f"No recorded model response for key {cache_key}")

    monkeypatch.setattr(a2a_app, "answer_question", missing_response)
    client = TestClient(a2a_app.app)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What does the policy say?"},
    )

    assert response.status_code == 503
    assert cache_key not in response.text
    assert "LLM_MODE=record" in response.json()["detail"]
    assert any(
        event["event_type"] == "chat.message.failed"
        for event in AuditStore().list_recent(20)
    )


def test_mcp_registers_only_answer_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "answer_with_policy",
        lambda question_id, question: {
            "status": "answered",
            "final_answer": {"question_id": question_id, "answer": question},
        },
    )

    async def invoke_tool():
        names = {tool.name for tool in await mcp_server.mcp.list_tools()}
        assert names == {"answer_question"}
        answer = await mcp_server.mcp.call_tool(
            "answer_question", {"question": "Can you cite evidence?"}
        )
        assert answer

    asyncio.run(invoke_tool())


def test_mcp_does_not_return_an_unreviewed_draft(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "answer_with_policy",
        lambda question_id, question: {
            "question_id": question_id,
            "status": "answered",
            "final_answer": {
                "question_id": question_id,
                "answer": "This answer is awaiting human review.",
                "in_scope": True,
                "policy_related": True,
                "citations": [],
            },
            "grounding_f1": 0.1,
            "human_review_required": True,
            "review_draft": {
                "answer": "MCP draft that must stay hidden.",
                "citations": [],
            },
        },
    )

    async def invoke_tool():
        result = await mcp_server.mcp.call_tool(
            "answer_question", {"question": "What does this policy require?"}
        )
        assert "MCP draft that must stay hidden." not in str(result)
        assert "review_session_id" in str(result)
        assert "pending_review" in str(result)

    asyncio.run(invoke_tool())
