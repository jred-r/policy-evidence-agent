from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import qa.llm.client as llm
from qa.domain.schemas import Citation, DraftedAnswer
from qa.workflow.scoring import faithfulness_judge


class FakeRouter:
    def __init__(self, response: dict[str, object] | Exception) -> None:
        self.response = response
        self.calls = 0

    def completion(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture(autouse=True)
def reset_llm_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("QA_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv("QA_MODELS_CONFIG", raising=False)
    monkeypatch.setenv("LLM_MODE", "replay")
    llm.set_clock_for_tests(lambda: datetime(2026, 1, 1, tzinfo=UTC))
    llm.set_router_for_tests(None)
    yield
    llm.set_clock_for_tests(llm._system_clock)
    llm.set_router_for_tests(None)


def _response(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}], "model": "fake"}


def test_replay_miss_does_not_call_router_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FakeRouter(RuntimeError("network must not be used"))
    llm.set_router_for_tests(router)
    monkeypatch.setattr(
        socket, "create_connection", lambda *args, **kwargs: pytest.fail("network call")
    )

    with pytest.raises(LookupError, match="No recorded model response"):
        llm.complete([{"role": "user", "content": "hello"}])

    assert router.calls == 0


def test_record_response_is_reused_in_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(_response('{"ok": true}'))
    llm.set_router_for_tests(router)
    monkeypatch.setenv("LLM_MODE", "record")
    request = [{"role": "user", "content": "hello"}]

    recorded = llm.complete(request, temperature=0)
    monkeypatch.setenv("LLM_MODE", "replay")
    replayed = llm.complete(request, temperature=0)

    assert recorded == replayed
    assert router.calls == 1


def test_scoped_record_mode_resets_to_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(_response('{"ok": true}'))
    llm.set_router_for_tests(router)

    with llm.override_llm_mode("record"):
        recorded = llm.complete([{"role": "user", "content": "record this"}])

    assert recorded["choices"][0]["message"]["content"] == '{"ok": true}'
    with pytest.raises(LookupError, match="No recorded model response"):
        llm.complete([{"role": "user", "content": "do not call live"}])
    assert router.calls == 1


def test_budget_counts_failed_attempts_and_rolls_over(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  - name: test-model\nbudget:\n  daily_soft_limit: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QA_MODELS_CONFIG", str(config))
    now = datetime(2026, 1, 1, 23, 59, tzinfo=UTC)
    llm.set_clock_for_tests(lambda: now)

    assert llm._consume_budget() == 1
    with pytest.raises(llm.BudgetExceeded):
        llm._consume_budget()
    assert llm.budget_count() == 1

    now += timedelta(minutes=2)
    llm.set_clock_for_tests(lambda: now)
    assert llm._consume_budget() == 1
    assert llm.budget_count() == 1


def test_failed_live_call_counts_against_daily_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    router = FakeRouter(RuntimeError("provider unavailable"))
    llm.set_router_for_tests(router)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        llm.complete([{"role": "user", "content": "hello"}])

    assert router.calls == 1
    assert llm.budget_count() == 1


def test_router_uses_yaml_fallback_order_and_deployment_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - name: openrouter-free\n    rpm: 20\n"
        "  - name: gemini-flash\n    rpm: 20\n"
        "router:\n  num_retries: 2\n  cooldown_time: 30\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QA_MODELS_CONFIG", str(config))
    import litellm

    captured: dict[str, object] = {}

    class CaptureRouter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(litellm, "Router", CaptureRouter)
    llm.set_router_for_tests(None)
    llm._get_router()

    assert captured["fallbacks"] == [
        {"openrouter-free": ["gemini-flash"]},
    ]
    assert captured["num_retries"] == 2
    assert [item["rpm"] for item in captured["model_list"]] == [20, 20]


def test_langfuse_and_otel_callbacks_follow_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")

    llm._enable_callbacks()

    assert set(litellm.callbacks) == {"langfuse_otel", "otel"}


def test_faithfulness_judge_uses_a_different_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_complete(messages, *, model=None, **params):
        captured["model"] = model
        return _response('{"score": 0.9}')

    monkeypatch.setattr(llm, "complete", fake_complete)
    answer = DraftedAnswer(
        question_id="Q1",
        answer="MFA is required.",
        confidence=0.9,
        citations=[Citation(passage_id="P1")],
    )

    assert faithfulness_judge(answer, "MFA is required.") == 0.9
    assert captured["model"] == "gemini-flash"


def test_record_and_replay_write_expected_trace_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm.set_router_for_tests(FakeRouter(_response('{"ok": true}')))
    request = [{"role": "user", "content": "trace me"}]

    monkeypatch.setenv("LLM_MODE", "record")
    llm.complete(request)
    monkeypatch.setenv("LLM_MODE", "replay")
    llm.complete(request)

    trace_rows = [
        json.loads(line)
        for line in Path(os.environ["QA_TRACE_PATH"]).read_text(encoding="utf-8").splitlines()
    ]
    assert trace_rows[0]["cache_hit"] is False
    assert trace_rows[1]["cache_hit"] is True
    assert {"model", "latency_ms", "request_count"} <= set(trace_rows[0])


def test_native_tool_response_validates_with_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = DraftedAnswer(
        question_id="Q1",
        answer="Access is reviewed.",
        confidence=0.8,
        citations=[Citation(passage_id="P1")],
    ).model_dump_json()
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"arguments": raw}}],
                }
            }
        ]
    }
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: response)

    result = llm.call_with_tools([], [], DraftedAnswer)

    assert result.question_id == "Q1"
    assert result.citations[0].passage_id == "P1"


def test_no_tool_support_uses_json_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = DraftedAnswer(
        question_id="Q1",
        answer="Access is reviewed.",
        confidence=0.8,
        citations=[Citation(passage_id="P1")],
    ).model_dump_json()
    calls = 0

    def complete(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider does not support tools")
        return _response(raw)

    monkeypatch.setattr(llm, "complete", complete)

    result = llm.call_with_tools([], [], DraftedAnswer)

    assert result.question_id == "Q1"
    assert calls == 2


def test_invalid_tool_json_gets_one_repair_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = DraftedAnswer(
        question_id="Q1",
        answer="Access is reviewed.",
        confidence=0.8,
        citations=[Citation(passage_id="P1")],
    ).model_dump_json()
    responses = iter([_response("{not-json"), _response(valid)])
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: next(responses))

    result = llm.call_with_tools([], [], DraftedAnswer)

    assert result.question_id == "Q1"
