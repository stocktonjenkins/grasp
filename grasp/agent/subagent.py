"""Sub-agent execution (§3.3, Figure 2): retrieve entities, select, read passages, decide.

Each sub-agent keeps a compact observation state instead of a message history, and
never revisits an entity or passage it has already seen.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from grasp.agent.schemas import EvaluationOutput, NodeSelectionOutput
from grasp.graph.backends import Candidate, Evidence, GraphBackend
from grasp.llm import TRANSIENT_RETRY, Backbone
from grasp.prompts import load_prompt

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2
CANDIDATE_ENTITIES = 5
NO_ANSWER = "No answer found."
FINAL_ITERATION_NOTE = (
    "\n== IMPORTANT ==\n"
    "This is the final retrieval iteration. You MUST provide your best "
    "answer based on all available evidence. If you cannot find a perfect "
    "answer, provide the closest answer supported by the evidence. "
    "Do not return an empty answer.\n"
)
NO_FINDINGS_YET = "Initial retrieval — no findings yet."


class Observation(BaseModel):
    """The sub-agent's observation state O_i, shown to the model at every call."""

    parent_question: str
    subquestion: str
    prior_answers: list[tuple[str, str]]
    step: int = 1
    visited_groups: list[dict] = Field(default_factory=list)
    missing_information: str = ""
    sufficient_evidence: bool = False

    def render(self) -> str:
        lines = [f"Parent question: {self.parent_question}", f"Subquestion: {self.subquestion}"]
        lines += [f"Prior answer #{i}: Q: {q} → A: {a}" for i, (q, a) in enumerate(self.prior_answers, 1)]
        lines += [
            f"Step: {self.step} of {MAX_ITERATIONS}",
            f"Sufficient evidence: {self.sufficient_evidence}",
            f"Missing information: {self.missing_information or NO_FINDINGS_YET}",
        ]
        if self.visited_groups:
            lines.append("Visited node groups:")
            lines += [f"  Group {i} [{', '.join(g['node_names'])}]: {g['findings']}" for i, g in enumerate(self.visited_groups, 1)]
        else:
            lines.append("Visited nodes: (none)")
        return "\n".join(lines)

    def visited_ids(self) -> str:
        if not self.visited_groups:
            return "none"
        return ", ".join(str(node_id) for group in self.visited_groups for node_id in group["node_ids"])


class SubAgentResult(BaseModel):
    answer: str
    evidence: list[Evidence]
    iterations: list[dict]


class SubAgent:
    def __init__(self, backbone: Backbone, temperature: float) -> None:
        self.backbone = backbone
        self.temperature = temperature

    def run(
        self,
        graph: GraphBackend,
        search_statement: str,
        keywords: list[str],
        config: RunnableConfig,
        parent_question: str = "",
        prior_answers: list[tuple[str, str]] | None = None,
    ) -> SubAgentResult:
        state = Observation(parent_question=parent_question, subquestion=search_statement, prior_answers=prior_answers or [])
        visited: set[int] = set()
        seen_evidence: set[int] = set()
        evidence: list[Evidence] = []
        iterations: list[dict] = []
        query, query_keywords, last_answer = search_statement, keywords, ""

        for iteration in range(MAX_ITERATIONS):
            selected, new_evidence = self.traverse(graph, query, query_keywords, search_statement, visited, seen_evidence, state, config)
            evidence += new_evidence
            if not new_evidence and not state.visited_groups:
                break
            is_final = iteration == MAX_ITERATIONS - 1
            decision = self.evaluate_evidence(state, new_evidence, is_final, config)
            state.step += 1
            state.visited_groups.append(
                {
                    "node_ids": [c.entity_id for c in selected],
                    "node_names": [c.entity_name for c in selected],
                    "findings": decision.node_findings,
                }
            )
            iterations.append(
                {
                    "query": query,
                    "keywords": query_keywords,
                    "selected": [c.entity_name for c in selected],
                    "evidence_ids": [e.id for e in new_evidence],
                    "action": decision.action,
                    "answer": decision.answer,
                }
            )
            if decision.action == "DONE":
                return SubAgentResult(answer=decision.answer, evidence=evidence, iterations=iterations)
            if is_final:
                return SubAgentResult(answer=decision.answer or last_answer or NO_ANSWER, evidence=evidence, iterations=iterations)
            last_answer = decision.answer or last_answer
            query, query_keywords = decision.new_search_statement or search_statement, decision.keywords
        return SubAgentResult(answer=last_answer or NO_ANSWER, evidence=evidence, iterations=iterations)

    def traverse(
        self,
        graph: GraphBackend,
        query: str,
        keywords: list[str],
        sub_question: str,
        visited: set[int],
        seen_evidence: set[int],
        state: Observation,
        config: RunnableConfig,
    ) -> tuple[list[Candidate], list[Evidence]]:
        candidates = [
            c for c in graph.candidates(query, keywords, k=CANDIDATE_ENTITIES, exclude_ids=list(visited)) if c.entity_id not in visited
        ]
        if not candidates:
            return [], []
        if len(candidates) == 1 and candidates[0].entity_id == -1:
            selected = candidates
        else:
            selected = self.select_nodes(query, sub_question, candidates, state, config)
        visited.update(c.entity_id for c in selected)
        new_evidence = graph.evidence([c.entity_id for c in selected], query, keywords, exclude_ids=list(seen_evidence))
        seen_evidence.update(e.id for e in new_evidence)
        return selected, new_evidence

    def select_nodes(
        self, query: str, sub_question: str, candidates: list[Candidate], state: Observation, config: RunnableConfig
    ) -> list[Candidate]:
        context = [
            f"Missing information: {state.missing_information or NO_FINDINGS_YET}",
            f"Already visited node IDs: {state.visited_ids()}",
        ]
        context += [f"Prior answer #{i}: Q: {q} → A: {a}" for i, (q, a) in enumerate(state.prior_answers, 1)]
        prompt = load_prompt(
            "node_selection",
            sub_question=sub_question,
            search_statement=query,
            candidates="\n".join(
                f"[ID: {c.entity_id}] {c.entity_name}" + (f" ({c.entity_type})" if c.entity_type else "") + f" (score: {c.score:.2f})"
                for c in candidates
            ),
            state_context="\n".join(context),
        )
        selector = self.backbone.chat(self.temperature).with_structured_output(NodeSelectionOutput).with_retry(**TRANSIENT_RETRY)
        by_id = {c.entity_id: c for c in candidates}
        try:
            chosen = selector.invoke([("user", prompt)], config=config).node_ids
        except Exception as error:  # noqa: BLE001
            logger.warning("Node selection failed (%s); using the top candidate.", error)
            return [candidates[0]]
        return [by_id[node_id] for node_id in chosen if node_id in by_id] or [candidates[0]]

    def evaluate_evidence(
        self, state: Observation, new_evidence: list[Evidence], is_final: bool, config: RunnableConfig
    ) -> EvaluationOutput:
        prompt = load_prompt(
            "evidence_evaluation",
            state_block=state.render(),
            new_evidence="\n".join(f"[ID: {e.id}] {e.text}" for e in new_evidence) or "(no new evidence retrieved this iteration)",
            final_iteration_note=FINAL_ITERATION_NOTE if is_final else "",
        )
        evaluator = self.backbone.chat(self.temperature).with_structured_output(EvaluationOutput).with_retry(**TRANSIENT_RETRY)
        try:
            decision: EvaluationOutput = evaluator.invoke([("user", prompt)], config=config)
        except Exception as error:  # noqa: BLE001
            logger.warning("Evidence evaluation failed (%s); returning an empty answer.", error)
            return EvaluationOutput(action="DONE", answer="", reasoning=f"Fallback: LLM error ({error}), returning empty answer.")
        state.missing_information = decision.missing_information or state.missing_information
        state.sufficient_evidence = decision.action == "DONE"
        return decision
