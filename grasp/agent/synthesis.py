"""Answer synthesis (§3.3): consolidate sub-question/answer pairs into one short answer."""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.runnables import RunnableConfig

from grasp.graph.backends import Evidence
from grasp.llm import TRANSIENT_RETRY, Backbone
from grasp.prompts import load_prompt

SynthesisMode = Literal["default", "fewshot"]
EVIDENCE_LIMIT = 20
EVIDENCE_HEADER = "Retrieved evidence (may contain the answer even when a step above reports it was not found):\n"


def extract_answer(response: str) -> str:
    match = re.search(r"[Ss]o the answer is[:\s]+(.+?)\.?\s*$", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


class Synthesizer:
    """`fewshot` prepends worked examples; `with_evidence` also passes the retrieved passages."""

    def __init__(self, backbone: Backbone, temperature: float, mode: SynthesisMode = "default", with_evidence: bool = False) -> None:
        self.backbone = backbone
        self.temperature = temperature
        self.mode = mode
        self.with_evidence = with_evidence

    def synthesize(
        self, question: str, history: list[tuple[str, str]], evidence: list[Evidence], config: RunnableConfig
    ) -> tuple[str, str]:
        research = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        passages = self.evidence_lines(evidence) if self.with_evidence else []
        if passages:
            research = f"{research}\n\n{EVIDENCE_HEADER}" + "\n".join(passages)
        prompt = f"Research:\n{research}\n\nQ: {question}\nA:"
        if self.mode == "fewshot":
            prompt = f"{load_prompt('synthesis_examples')}\n\n{prompt}"
        llm = self.backbone.chat(self.temperature).with_retry(**TRANSIENT_RETRY)
        content = llm.invoke([("system", load_prompt("synthesis_system")), ("user", prompt)], config=config).content
        if isinstance(content, list):
            content = content[0]["text"] if content else ""
        return extract_answer(content), content

    @staticmethod
    def evidence_lines(evidence: list[Evidence]) -> list[str]:
        seen: set[int] = set()
        lines: list[str] = []
        for unit in evidence:
            if not unit.text or unit.id in seen:
                continue
            seen.add(unit.id)
            lines.append(f"[ID: {unit.id}] {unit.text}")
            if len(lines) >= EVIDENCE_LIMIT:
                break
        return lines
