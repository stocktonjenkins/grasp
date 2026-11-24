from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from grasp.agent.synthesis import SynthesisMode
from grasp.datasets import Chunking
from grasp.graph.backends import GraphKind
from grasp.indexing.extraction import ExtractionStrategy
from grasp.llm import Provider


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraspSettings(Strict):
    graph: GraphKind = "proposition_passage"
    per_question: bool = False
    extraction: ExtractionStrategy = "batch-joint"
    batch_size: int = 10
    index_workers: int = 1
    synthesis: SynthesisMode = "default"
    synthesis_evidence: bool = False
    temperature: float = 0.2
    lexical_weight: float = 0.0  # Eq. 1 lambda; the paper's QA runs use 0


class BaselineSettings(Strict):
    variant: Literal["no_context", "with_context", "retriever", "decomposed", "self_ask"]
    index_path: Path | None = None
    top_k: int = 5
    global_retrieval: bool = False
    max_hops: int = 8


class RunConfig(Strict):
    dataset: str
    data: Path | None = None
    chunking: Chunking | None = None
    model: str = "gemini-3-flash-preview"
    provider: Provider = "gemini"
    base_url: str | None = None
    embedding_model: str = "gemini-embedding-001"
    embedding_provider: Literal["gemini", "huggingface"] = "gemini"
    embedding_dim: int = 3072
    parallel: int = 10
    neo4j_dir: Path | None = None
    output: Path | None = None
    grasp: GraspSettings | None = None
    baseline: BaselineSettings | None = None

    @model_validator(mode="after")
    def require_one_method(self) -> RunConfig:
        if (self.grasp is None) == (self.baseline is None):
            raise ValueError("A config needs exactly one of `grasp:` or `baseline:`.")
        if self.grasp is not None and self.neo4j_dir is None:
            raise ValueError("GRASP configs need `neo4j_dir`: where the graph is stored.")
        return self

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        config = cls.model_validate(yaml.safe_load(Path(path).read_text()))
        if config.output is None:
            config.output = Path("results") / Path(path).with_suffix(".json").relative_to(Path(path).parts[0])
        return config
