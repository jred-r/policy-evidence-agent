"""LiteLLM routing, offline replay, daily request budgets, and tracing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import yaml
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from ..storage.db import connection

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
LLMMode = Literal["live", "record", "replay", "auto"]
_mode_override: ContextVar[LLMMode | None] = ContextVar("llm_mode_override", default=None)


class BudgetExceeded(RuntimeError):
    """Raised when the configured UTC-day request budget is exhausted."""


class Router(Protocol):
    """Subset of LiteLLM Router used by this module."""

    def completion(self, **kwargs: Any) -> Any: ...


_router: Router | None = None
_dotenv_loaded = False


def _system_clock() -> datetime:
    return datetime.now(UTC)


_clock: Callable[[], datetime] = _system_clock


@contextmanager
def override_llm_mode(mode: Literal["record", "replay"]) -> Iterator[None]:
    """Temporarily select a mode for one workflow draft."""
    token = _mode_override.set(mode)
    try:
        yield
    finally:
        _mode_override.reset(token)


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load model and budget configuration from YAML."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        from dotenv import load_dotenv

        load_dotenv()
        _dotenv_loaded = True
    config_path = path or Path(os.getenv("QA_MODELS_CONFIG", "config/models.yaml"))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError(f"Invalid model configuration: {config_path}")
    if not raw["models"]:
        raise ValueError("At least one model deployment must be configured")
    return raw


def _get_router() -> Router:
    global _router
    if _router is None:
        from litellm import Router as LiteLLMRouter

        config = load_config()
        proxy = config.get("proxy", {})
        model_list = []
        for item in config["models"]:
            model_list.append(
                {
                    "model_name": item["name"],
                    "litellm_params": {
                        "model": f"openai/{item['name']}",
                        "api_base": proxy.get(
                            "api_base", "os.environ/LITELLM_API_BASE"
                        ),
                        "api_key": proxy.get(
                            "api_key", "os.environ/LITELLM_API_KEY"
                        ),
                        "timeout": int(item.get("timeout", 45)),
                    },
                    "rpm": int(item.get("rpm", 20)),
                }
            )
        router_config = config.get("router", {})
        fallbacks = [
            {model_list[index]["model_name"]: [entry["model_name"]]}
            for index in range(len(model_list) - 1)
            for entry in model_list[index + 1 : index + 2]
        ]
        _router = LiteLLMRouter(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=int(router_config.get("num_retries", 1)),
            cooldown_time=int(router_config.get("cooldown_time", 30)),
            allowed_fails=int(router_config.get("allowed_fails", 3)),
        )
    return _router


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _response_key(
    messages: Sequence[dict[str, Any]], params: dict[str, Any], model: str
) -> str:
    payload = {
        "messages": messages,
        "params": params,
        "tools": params.get("tools"),
        "model": model,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_replay(key: str) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT response FROM qa_llm_responses WHERE cache_key = %s", (key,)
        ).fetchone()
    return row["response"] if row else None


def _write_replay(key: str, response: dict[str, Any]) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO qa_llm_responses (cache_key, response, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response, created_at = EXCLUDED.created_at
            """,
            (key, Jsonb(response), _clock()),
        )


def _consume_budget() -> int:
    config = load_config()
    soft_limit = min(int(config.get("budget", {}).get("daily_soft_limit", 45)), 50)
    if soft_limit < 1:
        raise ValueError("The daily request limit must be positive")
    today = _clock().astimezone(UTC).date()
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO qa_llm_daily_budget (utc_day, request_count)
            VALUES (%s, 1)
            ON CONFLICT (utc_day) DO UPDATE
            SET request_count = qa_llm_daily_budget.request_count + 1
            WHERE qa_llm_daily_budget.request_count < %s
            RETURNING request_count
            """,
            (today, soft_limit),
        ).fetchone()
        if row is None:
            raise BudgetExceeded(f"Daily model request limit {soft_limit} exceeded")
        count = int(row["request_count"])
    return count


def _trace(model: str | None, cache_hit: bool, started: float, count: int | None) -> None:
    path = Path(os.getenv("QA_TRACE_PATH", ".qa/traces.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "model": model,
        "cache_hit": cache_hit,
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "request_count": count,
        "timestamp": _clock().isoformat(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_stable_json(event) + "\n")


def complete(
    messages: Sequence[dict[str, Any]], *, model: str | None = None, **params: Any
) -> dict[str, Any]:
    """Complete chat messages through LiteLLM under the selected operating mode."""
    mode = (_mode_override.get() or os.getenv("LLM_MODE", "replay")).lower()
    if mode not in {"live", "record", "replay", "auto"}:
        raise ValueError(f"Unsupported LLM_MODE: {mode}")

    config = load_config()
    deployment = model or config["models"][0]["name"]
    configured_models = {item["name"] for item in config["models"]}
    if deployment not in configured_models:
        raise ValueError(f"Model {deployment!r} is not listed in config/models.yaml")
    started = time.monotonic()
    key = _response_key(messages, params, deployment)
    cached = _read_replay(key) if mode in {"record", "replay", "auto"} else None
    if cached is not None:
        _trace(cached.get("model"), True, started, None)
        return cached
    if mode == "replay":
        raise LookupError(f"No recorded model response for key {key}")

    request_count = _consume_budget()
    router = _get_router()
    try:
        _enable_callbacks()
        response = router.completion(model=deployment, messages=list(messages), **params)
        data = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        if not isinstance(data, dict):
            raise TypeError("LiteLLM returned a non-object response")
        if mode in {"record", "auto"}:
            _write_replay(key, data)
        return data
    except Exception as error:
        logger.error(
            "Model request failed for %s (%s)", deployment, type(error).__name__
        )
        raise
    finally:
        _trace(deployment, False, started, request_count)


def _enable_callbacks() -> None:
    """Enable optional LiteLLM integrations from environment configuration."""
    import litellm

    callbacks = list(litellm.callbacks or [])
    if (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and "langfuse_otel" not in callbacks
    ):
        callbacks.append("langfuse_otel")
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and "otel" not in callbacks:
        callbacks.append("otel")
    litellm.callbacks = callbacks


def call_with_tools(
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]],
    result_model: type[T],
) -> T:
    """Call native tools when supported, otherwise request and validate strict JSON."""
    try:
        response = complete(messages, tools=tools, tool_choice="auto")
    except Exception as error:
        message = str(error).lower()
        unsupported = any(
            phrase in message
            for phrase in (
                "tool calling",
                "tool_calls",
                "does not support tools",
                "function calling",
            )
        )
        if not unsupported:
            raise
        response = _json_fallback(messages, tools, result_model)
        return response

    arguments = _native_arguments(response)
    if arguments is not None:
        return result_model.model_validate_json(arguments)
    content = _content(response)
    return _validate_with_one_repair(messages, tools, result_model, content)


def _json_fallback(
    messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]], result_model: type[T]
) -> T:
    instruction = (
        "Return only a JSON object matching this schema. Do not use markdown.\n"
        + _stable_json(result_model.model_json_schema())
        + "\nAvailable tool schemas:\n"
        + _stable_json(tools)
    )
    fallback_messages = [*messages, {"role": "system", "content": instruction}]
    response = complete(fallback_messages, response_format={"type": "json_object"})
    return _validate_with_one_repair(
        fallback_messages, tools, result_model, _content(response)
    )


def _validate_with_one_repair(
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]],
    result_model: type[T],
    content: str,
) -> T:
    try:
        return result_model.model_validate_json(content)
    except (ValueError, ValidationError) as validation_error:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Repair this as valid JSON for the requested schema. "
                    f"Errors: {validation_error}"
                ),
            },
        ]
        repaired = complete(repair_messages, response_format={"type": "json_object"})
        return result_model.model_validate_json(_content(repaired))


def _native_arguments(response: dict[str, Any]) -> str | None:
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    calls = message.get("tool_calls", [])
    if not calls:
        return None
    return calls[0]["function"]["arguments"]


def _content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Model response has no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Model response does not contain text content")
    return content


def budget_count(day: date | None = None) -> int:
    """Return the persisted request count for a UTC date."""
    selected = day or _clock().astimezone(UTC).date()
    with connection() as conn:
        row = conn.execute(
            "SELECT request_count FROM qa_llm_daily_budget WHERE utc_day = %s",
            (selected,),
        ).fetchone()
    return int(row["request_count"]) if row else 0


def set_clock_for_tests(clock: Callable[[], datetime]) -> None:
    """Set the clock used for deterministic unit tests."""
    global _clock
    _clock = clock


def set_router_for_tests(router: Router | None) -> None:
    """Set the router used for deterministic unit tests."""
    global _router
    _router = router
