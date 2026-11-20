"""Command-line tools for corpus validation, retrieval, evaluation, and answers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer

from .data.ingest import build_retrieval_baseline, validate_data
from .data.retrieve import PolicyRetriever
from .evaluation.runner import evaluate
from .storage.chat import ChatStore
from .workflow.answer import answer_question

app = typer.Typer(no_args_is_help=True)
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
def answer(question: str) -> None:
    """Answer one policy question."""
    result = answer_question(str(uuid.uuid4()), question)
    output = dict(result)
    if result.get("human_review_required"):
        draft = result.get("review_draft")
        if not draft:
            raise RuntimeError("Answer requires review but its draft is missing")
        output["review_session_id"] = ChatStore().create_review_session(
            str(result["question_id"]),
            question,
            str(draft["answer"]),
            draft.get("citations", []),
            result.get("grounding_f1"),
            "grounding_threshold",
        )
        output["status"] = "pending_review"
    output.pop("review_draft", None)
    typer.echo(json.dumps(output, indent=2, default=str))


@app.command()
def eval(output: Path = OUTPUT_PATH_OPTION) -> None:
    """Run replayable evaluation and write JSON and text results."""
    metrics = evaluate(output_dir=output)
    typer.echo(
        f"Evaluated {metrics['question_count']} answers; "
        f"answer-key F1={metrics['answer_key_f1']:.3f}"
    )


if __name__ == "__main__":
    app()
