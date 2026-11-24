"""GRASP inference (Figure 2): plan, run one sub-agent per sub-question, synthesize."""

from __future__ import annotations

from pydantic import BaseModel, Field

from grasp.agent.planning import Planner, Rewriter
from grasp.agent.subagent import SubAgent
from grasp.agent.synthesis import SynthesisMode, Synthesizer
from grasp.agent.validator import Validator, parse_candidates
from grasp.graph.backends import Evidence, GraphBackend
from grasp.llm import Backbone
from grasp.tokens import TokenCounter, TokenUsage


class AgentResult(BaseModel):
    answer: str
    rationale: str
    rational_plan: str
    usage: TokenUsage
    steps: list[dict] = Field(default_factory=list)


class Grasp:
    def __init__(
        self,
        backbone: Backbone,
        temperature: float = 0.2,
        synthesis: SynthesisMode = "default",
        synthesis_evidence: bool = False,
    ) -> None:
        self.planner = Planner(backbone, temperature)
        self.rewriter = Rewriter(backbone, temperature)
        self.subagent = SubAgent(backbone, temperature)
        self.validator = Validator(backbone, temperature, self.subagent)
        self.synthesizer = Synthesizer(backbone, temperature, synthesis, synthesis_evidence)

    def answer(self, question: str, graph: GraphBackend, counter: TokenCounter | None = None) -> AgentResult:
        counter = counter or TokenCounter()
        config = counter.config
        plan = self.planner.plan(question, config)
        sub_questions = [step.question for step in plan.steps]
        history: list[tuple[str, str]] = []
        evidence: list[Evidence] = []
        trace: list[dict] = []

        for i, sub_question in enumerate(sub_questions):
            rewrite = self.rewriter.rewrite(question, sub_questions, [a for _, a in history], sub_question, config)
            result = self.subagent.run(graph, rewrite.search_statement, rewrite.keywords, config, question, list(history))
            evidence += result.evidence
            step = {
                "sub_question": sub_question,
                "search_statement": rewrite.search_statement,
                "keywords": rewrite.keywords,
                "answer": result.answer,
                "iterations": result.iterations,
            }

            candidates = parse_candidates(result.answer)
            next_step = sub_questions[i + 1] if i + 1 < len(sub_questions) else ""
            if len(candidates) > 1 and f"#{i + 1}" in next_step:
                validation = self.validator.choose(graph, question, rewrite.search_statement, next_step, candidates, config)
                evidence += validation.evidence
                step["validation"] = {"chosen": validation.chosen, "probes": validation.probes}
                history.append((rewrite.search_statement, validation.chosen))
            else:
                history.append((rewrite.search_statement, result.answer))
            trace.append(step)

        answer, rationale = self.synthesizer.synthesize(question, history, evidence, config)
        return AgentResult(answer=answer, rationale=rationale, rational_plan=plan.rational_plan, usage=counter.usage, steps=trace)
