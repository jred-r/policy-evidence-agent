"""Replayable answer evaluation and result serialization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..data.ingest import validate_data
from ..data.retrieve import PolicyRetriever
from ..domain.schemas import DataSet
from ..llm.client import budget_count, load_config
from ..workflow.draft import draft_answers
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
    """Evaluate one-pass answers, then write JSON and text results."""
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
    before_budget = budget_count()
    drafted_answers = draft_answers(data.questions, search, batch_size=5)
    source_text = {
        passage.id: passage.text
        for document in data.policies
        for passage in document.passages
    }
    key_by_id = {entry.question_id: entry for entry in data.answer_key}
    results = []
    for answer in drafted_answers:
        key = key_by_id.get(answer.question_id)
        results.append(
            {
                **answer.model_dump(),
                "grounding_f1": grounding_f1(answer, source_text),
                "answer_key_f1": token_f1(answer.answer, key.answer) if key else 0.0,
                "status": "answered" if answer.in_scope else "off_topic",
            }
        )
    answers = drafted_answers
    grounding_scores = [float(row["grounding_f1"]) for row in results]
    out_of_scope = sum(not answer.in_scope for answer in answers)
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
        "answer_key_f1": answer_key_f1(answers, data.answer_key),
        "citation_correctness": citation_correctness(answers, data.answer_key),
        "mean_grounding_f1": (
            sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
        ),
        "out_of_scope_percent": 100 * out_of_scope / len(results) if results else 0.0,
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
        ("Answer-key F1", f"{metrics['answer_key_f1']:.3f}"),
        ("Mean grounding F1", f"{metrics['mean_grounding_f1']:.3f}"),
        ("Citation correctness", f"{metrics['citation_correctness']:.3f}"),
        ("Out-of-scope percent", f"{metrics['out_of_scope_percent']:.1f}%"),
        ("Requests per answer", f"{metrics['requests_per_answer']:.2f}"),
    ]
    width = max(len(headers[0]), *(len(row[0]) for row in rows))
    return "\n".join(
        [f"{headers[0]:<{width}} | {headers[1]}", f"{'-' * width}-+-{'-' * 8}"]
        + [f"{name:<{width}} | {value}" for name, value in rows]
    ) + "\n"
