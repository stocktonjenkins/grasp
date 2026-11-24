"""Dependency-aware planning (§3.2) and per-step search-statement rewriting (§3.3)."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from grasp.agent.schemas import Plan, RewriteResult
from grasp.llm import TRANSIENT_RETRY, Backbone
from grasp.prompts import load_prompt


class Planner:
    def __init__(self, backbone: Backbone, temperature: float) -> None:
        self.backbone = backbone
        self.temperature = temperature

    def plan(self, question: str, config: RunnableConfig) -> Plan:
        planner = self.backbone.chat(self.temperature).with_structured_output(Plan).with_retry(**TRANSIENT_RETRY)
        return planner.invoke([("system", load_prompt("planner")), ("user", question)], config=config)


class Rewriter:
    """Resolves #N references and turns a sub-question into a search statement plus keywords."""

    def __init__(self, backbone: Backbone, temperature: float) -> None:
        self.backbone = backbone
        self.temperature = temperature

    def rewrite(self, question: str, plan: list[str], answers: list[str], sub_question: str, config: RunnableConfig) -> RewriteResult:
        prompt = load_prompt(
            "rewriter",
            original_question=question,
            rational_plan="\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan)),
            context_history="\n".join(f"Step {i + 1} Answer: {answer}" for i, answer in enumerate(answers)),
            current_sub_question=sub_question,
        )
        rewriter = self.backbone.chat(self.temperature).with_structured_output(RewriteResult).with_retry(**TRANSIENT_RETRY)
        return rewriter.invoke([("user", prompt)], config=config)
