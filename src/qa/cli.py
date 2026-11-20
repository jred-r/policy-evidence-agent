"""Command-line tools for corpus validation, retrieval, evaluation, and review."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .evaluation import evaluate
from .graph import run_workflow
from .ingest import build_retrieval_baseline, validate_data
from .retrieve import PolicyRetriever
from .review import ReviewStore

app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(review_app, name="review")
DATA_PATH_OPTION = typer.Option(None, "--path", help="Corpus directory")
OUTPUT_PATH_OPTION = typer.Option(Path("results"), "--output")


@app.command("validate-data")
def validate_data_command(
    path: Path | None = DATA_PATH_OPTION
) -> None:
    """Validate question, policy, and answer-key data."""
    dataset = validate_data(path)
    typer.echo(
        f"Validated {len(dataset.questions)} questions, "
        f"{len(dataset.policies)} policies, and {len(dataset.answer_key)} answers."
    )


@app.command()
def ingest(k: int = typer.Option(5, min=1)) -> None:
    """Ingest policy passages and report retrieval recall@k."""
    dataset = validate_data()
    retriever = PolicyRetriever()
    score = build_retrieval_baseline(retriever, dataset, k)
    typer.echo(f"recall@{k}: {score:.3f}")


@app.command()
def answer(question_id: str) -> None:
    """Answer one question from the validated corpus."""
    dataset = validate_data()
    question = next(
        (item for item in dataset.questions if item.id == question_id), None
    )
    if question is None:
        raise typer.BadParameter(f"Unknown question ID: {question_id}")
    result = run_workflow(question.id, question.text, thread_id=f"cli-{question.id}")
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def eval(output: Path = OUTPUT_PATH_OPTION) -> None:
    """Run replayable evaluation and write JSON and text results."""
    metrics = evaluate(output_dir=output)
    typer.echo(
        f"Evaluated {metrics['question_count']} answers; "
        f"answer-key F1={metrics['answer_key_f1']:.3f}"
    )


@review_app.command("list")
def list_review_queue() -> None:
    """List pending human-review items."""
    typer.echo(json.dumps(ReviewStore().list_pending(), indent=2))


@review_app.command("approve")
def approve(
    question_id: str,
    edited_answer: str | None = typer.Option(None, "--edit"),
) -> None:
    """Approve a queued answer, optionally with an edited response."""
    result = ReviewStore().decide(question_id, approved=True, edited_answer=edited_answer)
    typer.echo(json.dumps(result, indent=2))


@review_app.command("reject")
def reject(question_id: str) -> None:
    """Reject a queued answer."""
    result = ReviewStore().decide(question_id, approved=False)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
