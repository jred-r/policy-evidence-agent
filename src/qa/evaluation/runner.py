"""Replayable answer evaluation and result serialization."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..data.ingest import validate_data
from ..data.retrieve import PolicyRetriever
from ..domain.schemas import DataSet, DraftedAnswer
from ..llm.client import budget_count, load_config
from ..storage.db import database_url
from ..storage.review import ReviewStore
from ..workflow.draft import draft_answers
from ..workflow.graph import create_workflow
from ..workflow.scoring import (
    answer_key_f1,
    citation_correctness,
    faithfulness_judge,
    grounding_f1,
    token_f1,
)


def evaluate(
    dataset: DataSet | None = None,
    retriever: PolicyRetriever | None = None,
    output_dir: Path = Path("results"),
) -> dict[str, Any]:
    """Evaluate first drafts and durable retries, then write JSON and text results."""
    data = dataset or validate_data()
    search = retriever or PolicyRetriever()
    search.ingest(data.policies)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_trace = os.environ.get("QA_TRACE_PATH")
    with tempfile.TemporaryDirectory(prefix="qa-eval-") as trace_dir:
        os.environ["QA_TRACE_PATH"] = str(Path(trace_dir) / "trace.jsonl")
        try:
            metric_data = _evaluate(data, search, output_dir)
        finally:
            if previous_trace is None:
                os.environ.pop("QA_TRACE_PATH", None)
            else:
                os.environ["QA_TRACE_PATH"] = previous_trace
    return metric_data


def _evaluate(
    data: DataSet, search: PolicyRetriever, output_dir: Path
) -> dict[str, Any]:
    from langgraph.checkpoint.postgres import PostgresSaver

    before_budget = budget_count()
    initial_answers = draft_answers(data.questions, search, batch_size=5)
    source_text = {
        passage.id: passage.text
        for document in data.policies
        for passage in document.passages
    }
    key_by_id = {entry.question_id: entry for entry in data.answer_key}
    questions_by_id = {question.id: question for question in data.questions}
    initial_by_id = {answer.question_id: answer for answer in initial_answers}
    initial_scores = {
        answer.question_id: grounding_f1(answer, source_text)
        for answer in initial_answers
    }
    retry_questions = [
        questions_by_id[question_id]
        for question_id, score in initial_scores.items()
        if score < 0.95
    ]
    retry_by_id = {
        answer.question_id: answer
        for answer in draft_answers(retry_questions, search, batch_size=5)
    } if retry_questions else {}

    attempts: dict[str, int] = {}

    def choose_draft(question: Any, passages: Any) -> Any:
        count = attempts.get(question.id, 0)
        attempts[question.id] = count + 1
        if count == 0:
            return initial_by_id[question.id]
        return retry_by_id[question.id]

    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        review_store = ReviewStore()
        results = []
        for question in data.questions:
            workflow = create_workflow(
                retriever=search,
                review_store=review_store,
                checkpointer=saver,
                draft_function=choose_draft,
            )
            result = workflow.invoke(
                {"question_id": question.id, "question": question.text},
                config={"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}},
            )
            final_raw = result.get("final_answer") or result.get("best_answer")
            answer = initial_by_id[question.id]
            if final_raw is not None:
                answer = DraftedAnswer.model_validate(final_raw)
            score = grounding_f1(answer, source_text)
            key = key_by_id.get(question.id)
            results.append(
                {
                    **answer.model_dump(),
                    "first_attempt_grounding_f1": initial_scores[question.id],
                    "grounding_f1": score,
                    "answer_key_f1": token_f1(answer.answer, key.answer) if key else 0.0,
                    "attempts": result.get("attempts", 1),
                    "status": result.get("status", "unknown"),
                }
            )

    answers = [
        DraftedAnswer.model_validate(
            {key: row[key] for key in ("question_id", "answer", "confidence", "citations")}
        )
        for row in results
    ]
    grounding_scores = [float(row["grounding_f1"]) for row in results]
    first_passes = sum(score >= 0.95 for score in initial_scores.values())
    escalated = sum(row["status"] == "review" for row in results)
    judge_scores: list[float] = []
    if os.getenv("QA_ENABLE_JUDGE", "").lower() in {"1", "true", "yes"}:
        config = load_config()
        for answer in answers:
            cited = " ".join(
                source_text[citation.passage_id]
                for citation in answer.citations
                if citation.passage_id in source_text
            )
            judge_scores.append(
                faithfulness_judge(
                    answer,
                    cited,
                    judge_model=config.get("judge_model"),
                    generator_model=config["models"][0]["name"],
                )
            )
    trace_path = Path(os.environ["QA_TRACE_PATH"])
    trace_events = (
        len(trace_path.read_text(encoding="utf-8").splitlines())
        if trace_path.exists()
        else 0
    )
    live_requests = max(0, budget_count() - before_budget)
    metric_data = {
        "question_count": len(results),
        "first_attempt_pass_rate": first_passes / len(results) if results else 0.0,
        "f1_after_retries": answer_key_f1(answers, data.answer_key),
        "answer_key_f1": answer_key_f1(answers, data.answer_key),
        "citation_correctness": citation_correctness(answers, data.answer_key),
        "mean_grounding_f1": (
            sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
        ),
        "escalated_percent": (
            100 * escalated / len(results)
            if results
            else 0.0
        ),
        "retry_percent": (
            100 * sum(row["attempts"] > 1 for row in results) / len(results)
            if results
            else 0.0
        ),
        "requests_per_answer": trace_events / len(results) if results else 0.0,
        "live_requests_per_answer": live_requests / len(results) if results else 0.0,
        "mean_judge_faithfulness": (
            sum(judge_scores) / len(judge_scores) if judge_scores else None
        ),
        "answers": results,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metric_data, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(_summary(metric_data), encoding="utf-8")
    return metric_data


def _summary(metrics: dict[str, Any]) -> str:
    headers = ("Metric", "Value")
    rows = [
        ("Questions", str(metrics["question_count"])),
        ("First-attempt pass rate", f"{metrics['first_attempt_pass_rate']:.3f}"),
        ("Answer-key F1", f"{metrics['answer_key_f1']:.3f}"),
        ("F1 after retries", f"{metrics['f1_after_retries']:.3f}"),
        ("Mean grounding F1", f"{metrics['mean_grounding_f1']:.3f}"),
        ("Citation correctness", f"{metrics['citation_correctness']:.3f}"),
        ("Escalated percent", f"{metrics['escalated_percent']:.1f}%"),
        ("Retry percent", f"{metrics['retry_percent']:.1f}%"),
        ("Requests per answer", f"{metrics['requests_per_answer']:.2f}"),
    ]
    width = max(len(headers[0]), *(len(row[0]) for row in rows))
    return "\n".join(
        [f"{headers[0]:<{width}} | {headers[1]}", f"{'-' * width}-+-{'-' * 8}"]
        + [f"{name:<{width}} | {value}" for name, value in rows]
    ) + "\n"
