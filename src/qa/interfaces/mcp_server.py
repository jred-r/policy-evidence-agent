"""MCP tool for answering policy questions."""

from __future__ import annotations

import uuid

from mcp.server.fastmcp import FastMCP

from ..storage.chat import ChatStore
from ..workflow.answer import answer_question as answer_with_policy

mcp = FastMCP("policy-evidence-agent")


@mcp.tool()
def answer_question(question: str) -> dict[str, object]:
    """Answer one policy question with citations."""
    result = answer_with_policy(str(uuid.uuid4()), question)
    public_result = dict(result)
    if result.get("human_review_required"):
        draft = result.get("review_draft")
        if not draft:
            raise RuntimeError("Answer requires review but its draft is missing")
        public_result["review_session_id"] = ChatStore().create_review_session(
            str(result["question_id"]),
            question,
            str(draft["answer"]),
            draft.get("citations", []),
            result.get("grounding_f1"),
            "grounding_threshold",
        )
        public_result["status"] = "pending_review"
    public_result.pop("review_draft", None)
    return public_result


def main() -> None:
    """Start the MCP stdio server."""
    mcp.run()


if __name__ == "__main__":
    main()
