"""Disambiguates a sub-answer that lists several candidates by probing the next step with each."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from grasp.agent.subagent import SubAgent, SubAgentResult
from grasp.graph.backends import Evidence, GraphBackend
from grasp.llm import TRANSIENT_RETRY, Backbone

_CANDIDATES = re.compile(r"CANDIDATES:\s*(.+)", re.IGNORECASE | re.DOTALL)
_NUMBERED_ITEM = re.compile(r"\d+\)\s*(.+?)(?=\s*\d+\)|$)")
_CITATION = re.compile(r"\s*\[ID:[^\]]*\]")

JUDGE_SYSTEM = "You select the best candidate based on evidence. Be concise."


def parse_candidates(answer: str) -> list[str]:
    """`CANDIDATES: 1) A [ID: 4] 2) B [ID: 9]` -> ["A", "B"]; anything else -> [answer]."""
    match = _CANDIDATES.search(answer)
    if not match:
        return [answer]
    items = _NUMBERED_ITEM.findall(match.group(1).strip())
    candidates = [c for c in (_CITATION.sub("", item).strip() for item in items) if c]
    return candidates or [answer]


class Validation(BaseModel):
    chosen: str
    evidence: list[Evidence]
    probes: list[dict]


class Validator:
    def __init__(self, backbone: Backbone, temperature: float, subagent: SubAgent) -> None:
        self.backbone = backbone
        self.temperature = temperature
        self.subagent = subagent

    def choose(
        self, graph: GraphBackend, question: str, sub_question: str, next_step: str, candidates: list[str], config: RunnableConfig
    ) -> Validation:
        def probe(candidate: str) -> tuple[str, SubAgentResult]:
            return candidate, self.subagent.run(graph, re.sub(r"#\d+", candidate, next_step), candidate.split(), config)

        with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as pool:
            probes = [future.result() for future in as_completed([pool.submit(probe, c) for c in candidates])]

        chosen = self.judge_candidates(question, sub_question, next_step, probes, config)
        winner = next((result for candidate, result in probes if candidate == chosen), probes[0][1])
        return Validation(chosen=chosen, evidence=winner.evidence, probes=[{"candidate": c, "answer": r.answer} for c, r in probes])

    def judge_candidates(
        self, question: str, sub_question: str, next_step: str, probes: list[tuple[str, SubAgentResult]], config: RunnableConfig
    ) -> str:
        blocks = [f"[{i}] Candidate: {candidate}\n    Evidence: {result.answer}" for i, (candidate, result) in enumerate(probes)]
        prompt = (
            f"Original question: {question}\n"
            f"Subquestion that produced candidates: {sub_question}\n"
            f"Validation question: {next_step}\n\n"
            + "\n---\n".join(blocks)
            + "\n\nPick the candidate whose validation SUCCEEDED with evidence. "
            "If multiple succeeded, pick the one with the strongest evidence. "
            "If none succeeded, pick the most plausible.\n\n"
            "Respond with exactly one line:\n"
            "CHOSEN: <candidate name>"
        )
        judge = self.backbone.chat(self.temperature).with_retry(**TRANSIENT_RETRY)
        content = judge.invoke([("system", JUDGE_SYSTEM), ("user", prompt)], config=config).content
        if isinstance(content, list):
            content = next((part["text"] for part in content if isinstance(part, dict) and "text" in part), "")
        for line in (content or "").strip().split("\n"):
            if line.startswith("CHOSEN:"):
                return line[len("CHOSEN:") :].strip()
        return ""
