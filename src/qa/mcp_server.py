"""MCP tools for answering and reviewing policy questions."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .graph import run_workflow
from .review import ReviewStore

mcp = FastMCP("policy-evidence-agent")


@mcp.tool()
def answer_question(question_id: str, question: str) -> dict[str, object]:
    """Answer one security assessment question with policy citations."""
    result = run_workflow(
        question_id,
        question,
        thread_id=f"mcp-{question_id}",
    )
    return dict(result)


@mcp.tool()
def list_review_queue() -> list[dict[str, object]]:
    """List answers waiting for human review."""
    return ReviewStore().list_pending()


@mcp.tool()
def approve_answer(
    question_id: str, approved: bool, edited_answer: str | None = None
) -> dict[str, object]:
    """Approve, reject, or edit a queued answer."""
    return ReviewStore().decide(question_id, approved, edited_answer)


def main() -> None:
    """Start the MCP stdio server."""
    mcp.run()


if __name__ == "__main__":
    main()
