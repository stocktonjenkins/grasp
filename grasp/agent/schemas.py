"""Structured-output schemas. Class names, docstrings and field descriptions are sent to
the model as JSON schema, so they are part of the prompt and must not be edited."""

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """A single subquestion in the plan."""

    question: str


class Plan(BaseModel):
    """The plan of action to follow."""

    rational_plan: str = Field()
    steps: list[SubQuestion] = Field()


class RewriteResult(BaseModel):
    """Output of the question rewriter: a search statement plus fulltext keywords."""

    search_statement: str = Field()
    keywords: list[str] = Field()


class NodeSelectionOutput(BaseModel):
    node_ids: list[int] = Field()
    reasoning: str


class EvaluationOutput(BaseModel):
    """Step 4: Combined evidence evaluation."""

    action: Literal["DONE", "QUERY_AGAIN"]
    answer: str = Field(default="")
    supporting_prop_ids: list[int] = Field(default_factory=list)
    new_search_statement: str | None = Field(default=None)
    keywords: list[str] = Field(default_factory=list)
    node_findings: str = Field(
        default="",
        description=("1-2 sentence summary of what was learned from the nodes visited in this iteration, relative to the subquestion."),
    )
    missing_information: str = Field(
        default="",
        description="What information is still needed, or 'resolved' if DONE.",
    )
