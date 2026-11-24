"""Non-agentic baselines (Table 6): No Context, Full Context, RAG, Q-Dec + RAG and Self-Ask + RAG."""

from __future__ import annotations

import json
import re
from typing import Callable, Literal

import numpy as np

from grasp.baselines.dense import DenseIndex
from grasp.datasets import Question
from grasp.embeddings import Embedder
from grasp.llm import chat_model
from grasp.methods import Prediction
from grasp.prompts import load_prompt
from grasp.tokens import TokenCounter

Variant = Literal["no_context", "with_context", "retriever", "decomposed"]
FINAL_MARKER = "So the final answer is:"


def extract_answer(text: str) -> str:
    match = re.search(r"[Ss]o the answer is:\s*(.+?)\.?\s*$", text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        for _ in range(2):
            if answer and answer[-1] in ".?!":
                answer = answer[:-1]
        return answer
    return text[:-1].strip() if text.endswith((".", "?", "!")) else text.strip()


class DenseRetrieval:
    def __init__(self, model: str, embedder: Embedder, index: Callable[[], DenseIndex], top_k: int) -> None:
        self.llm = chat_model(model, "gemini", 0.0, thinking_level="minimal")
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def query_vector(self, text: str) -> np.ndarray:
        vector = np.array(self.embedder.query(text), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


class Vanilla(DenseRetrieval):
    def __init__(
        self,
        variant: Variant,
        model: str,
        embedder: Embedder,
        index: Callable[[], DenseIndex],
        top_k: int = 5,
        global_retrieval: bool = False,
    ) -> None:
        super().__init__(model, embedder, index, top_k)
        self.variant = variant
        self.global_retrieval = global_retrieval

    def answer(self, question: Question) -> Prediction:
        counter, decompose_counter = TokenCounter(), TokenCounter()
        if self.variant == "no_context":
            system, context = load_prompt("baseline_no_context"), None
        elif self.variant == "with_context":
            system, context = load_prompt("baseline_with_context"), [p.text for p in question.passages]
        elif self.variant == "retriever":
            system = load_prompt("baseline_with_context")
            context = self.index().retrieve(
                self.query_vector(question.question), question.id, self.top_k, global_retrieval=self.global_retrieval
            )
        else:
            system = load_prompt("baseline_with_context")
            texts: dict[str, None] = {}
            for sub_question in self.decompose(question.question, decompose_counter):
                for text in self.index().retrieve(self.query_vector(sub_question), question.id, self.top_k, global_retrieval=True):
                    texts.setdefault(text, None)
            context = list(texts)
        user = (
            f"Question: {question.question}"
            if context is None
            else "\n".join(t.strip() for t in context if t.strip()) + f"\n\nQuestion: {question.question}"
        )
        response = self.llm.invoke([("system", system), ("user", user)], config=counter.config)
        return Prediction(answer=extract_answer(response.text), usage=counter.usage, construction_usage=decompose_counter.usage)

    def decompose(self, question: str, counter: TokenCounter) -> list[str]:
        text = self.llm.invoke(
            [("system", load_prompt("baseline_decompose")), ("user", f"Question: {question}")], config=counter.config
        ).text.strip()
        if text.startswith("```"):
            text = re.sub(r"\n?```$", "", re.sub(r"^```[a-z]*\n?", "", text)).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [question]
        return parsed if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed) else [question]


class SelfAsk(DenseRetrieval):
    """Self-Ask (Press et al., 2023) with per-question dense retrieval in place of web search."""

    def __init__(self, model: str, embedder: Embedder, index: Callable[[], DenseIndex], top_k: int = 5, max_hops: int = 8) -> None:
        super().__init__(model, embedder, index, top_k)
        self.max_hops = max_hops

    def answer(self, question: Question) -> Prediction:
        counter = TokenCounter()
        chain = load_prompt("self_ask_prefix", question=question.question)
        for _ in range(self.max_hops):
            generated = self.llm.invoke([("user", chain)], stop=["Intermediate answer:"], config=counter.config).text
            chain += generated
            follow_up = parse_follow_up(generated)
            if follow_up is None:
                break
            passages = self.index().retrieve(self.query_vector(follow_up), question.id, self.top_k)
            chain += "Intermediate answer: " + self.intermediate_answer(follow_up, passages, counter) + "\n"
        if FINAL_MARKER.lower() not in chain.lower():
            chain += f"\n{FINAL_MARKER}"
            chain += self.llm.invoke([("user", chain)], stop=["\n"], config=counter.config).text
        return Prediction(answer=parse_final_answer(chain), usage=counter.usage, rationale=chain)

    def intermediate_answer(self, sub_question: str, passages: list[str], counter: TokenCounter) -> str:
        if not passages:
            return "I don't know."
        user = "Context:\n" + "\n\n".join(passages) + f"\n\nQuestion: {sub_question}"
        return self.llm.invoke([("system", load_prompt("self_ask_extract")), ("user", user)], config=counter.config).text.strip()


def parse_follow_up(text: str) -> str | None:
    for line in reversed(text.strip().splitlines()):
        match = re.match(r"[Ff]ollow\s*[Uu]p:\s*(.+)", line.strip())
        if match:
            return match.group(1).strip()
    return None


def parse_final_answer(chain: str) -> str:
    """Text after the last "So the final answer is:" — which may be a few-shot exemplar's
    if the model never answered, as in the paper's Self-Ask runs."""
    start = chain.lower().rfind(FINAL_MARKER.lower())
    if start == -1:
        return chain.strip()
    return chain[start + len(FINAL_MARKER) :].strip().splitlines()[0].strip().rstrip(".")
