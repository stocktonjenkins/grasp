"""Question answerers built from a run config: GRASP itself and the Table 6 baselines."""

from __future__ import annotations

import time
from typing import Protocol

from neo4j import Driver
from pydantic import BaseModel, Field

from grasp.agent.grasp import Grasp
from grasp.config import GraspSettings
from grasp.datasets import Question
from grasp.embeddings import Embedder
from grasp.graph.backends import BACKENDS
from grasp.graph.store import delete_question_graph, indexed_question, mark_indexed, question_graph_exists
from grasp.indexing.extraction import Extractor
from grasp.indexing.indexer import Indexer
from grasp.llm import Backbone
from grasp.tokens import TokenCounter, TokenUsage


class Prediction(BaseModel):
    answer: str
    usage: TokenUsage
    rationale: str = ""
    rational_plan: str = ""
    construction_usage: TokenUsage = Field(default_factory=TokenUsage)
    construction_seconds: float = 0.0
    trace: list[dict] = Field(default_factory=list)
    error: str | None = None


class Method(Protocol):
    def answer(self, question: Question) -> Prediction: ...


class GraspMethod:
    """Answers over the shared corpus graph, or over a graph built for each question (LongBench)."""

    def __init__(self, settings: GraspSettings, backbone: Backbone, embedder: Embedder, driver: Driver, embedding_dim: int) -> None:
        self.settings = settings
        self.driver = driver
        self.embedder = embedder
        self.agent = Grasp(backbone, settings.temperature, settings.synthesis, settings.synthesis_evidence)
        self.indexer = Indexer(driver, Extractor(settings.extraction, backbone, settings.batch_size), embedder, embedding_dim)

    def answer(self, question: Question) -> Prediction:
        scope = question.id if self.settings.per_question else None
        construction, seconds = self.question_graph(question) if scope is not None else (TokenUsage(), 0.0)

        graph = BACKENDS[self.settings.graph](self.driver, self.embedder, question_id=scope, lexical_weight=self.settings.lexical_weight)
        counter = TokenCounter()
        try:
            result = self.agent.answer(question.question, graph, counter)
        except Exception as error:  # noqa: BLE001
            return Prediction(
                answer="", usage=counter.usage, construction_usage=construction, construction_seconds=seconds, error=str(error)
            )
        return Prediction(
            answer=result.answer,
            usage=result.usage,
            rationale=result.rationale,
            rational_plan=result.rational_plan,
            construction_usage=construction,
            construction_seconds=seconds,
            trace=result.steps,
        )

    def question_graph(self, question: Question) -> tuple[TokenUsage, float]:
        """Build the question's own graph once and keep it; a re-run reuses it and reports what it cost."""
        if (built := indexed_question(self.driver, question.id)) is not None:
            return built
        if question_graph_exists(self.driver, question.id):
            delete_question_graph(self.driver, question.id)
        start = time.time()
        usage = self.indexer.index(question.indexable_passages(), question_id=question.id, workers=self.settings.index_workers)
        seconds = time.time() - start
        mark_indexed(self.driver, question.id, usage, seconds)
        return usage, seconds
