from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from neo4j import Driver
from neo4j.exceptions import TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from grasp.embeddings import Embedder
from grasp.graph.store import GraphWriter, ensure_passage_vector_index, ensure_schema, passage_exists
from grasp.indexing.extraction import Extractor
from grasp.tokens import TokenUsage

deadlock_retry = retry(
    retry=retry_if_exception_type(TransientError),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)


def embed_passages(driver: Driver, embedder: Embedder, embedding_dim: int, batch_size: int = 100) -> int:
    """Embed whole passages for the dense-passage ablation (Figure 5, "DPR (no graph)")."""
    ensure_passage_vector_index(driver, embedding_dim)
    with driver.session() as session:
        rows = [
            (r["id"], r["text"])
            for r in session.run("MATCH (p:Passage) WHERE p.embedding IS NULL AND p.text IS NOT NULL RETURN id(p) AS id, p.text AS text")
        ]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embedder.documents([text for _, text in batch])
        with driver.session() as session:
            session.run(
                "UNWIND $rows AS row MATCH (p:Passage) WHERE id(p) = row.id SET p.embedding = row.embedding",
                rows=[{"id": node_id, "embedding": vector} for (node_id, _), vector in zip(batch, vectors)],
            ).consume()
    return len(rows)


class Indexer:
    """Builds the proposition graph: extract each passage, then write and embed it."""

    def __init__(self, driver: Driver, extractor: Extractor, embedder: Embedder, embedding_dim: int) -> None:
        self.driver = driver
        self.extractor = extractor
        self.writer = GraphWriter(driver, embedder)
        self.embedding_dim = embedding_dim
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def index(
        self,
        passages: list[tuple[str, str]],
        *,
        question_id: str | None = None,
        skip_existing: bool = False,
        workers: int = 1,
    ) -> TokenUsage:
        with self._schema_lock:
            if not self._schema_ready:
                ensure_schema(self.driver, self.embedding_dim)
                self._schema_ready = True
        numbered = [(i, title, text.strip()) for i, (title, text) in enumerate(passages)]
        if workers > 1:
            return self.index_concurrently(numbered, question_id, skip_existing, workers)
        usage = TokenUsage()
        size = self.extractor.batch_size
        for start in range(0, len(numbered), size):
            usage += self.index_batch(numbered[start : start + size], question_id, skip_existing)
        return usage

    def index_batch(self, batch: list[tuple[int, str, str]], question_id: str | None, skip_existing: bool) -> TokenUsage:
        todo = [(i, title, text) for i, title, text in batch if text and not (skip_existing and passage_exists(self.driver, text))]
        if not todo:
            return TokenUsage()
        extractions, usage = self.extractor.extract([(title, text) for _, title, text in todo])
        for (i, _, text), extraction in zip(todo, extractions):
            self.writer.write(text, i, extraction, question_id)
        return usage

    def index_concurrently(
        self, numbered: list[tuple[int, str, str]], question_id: str | None, skip_existing: bool, workers: int
    ) -> TokenUsage:
        usage = TokenUsage()
        lock = threading.Lock()
        index_one = deadlock_retry(self.index_batch)

        def work(item: tuple[int, str, str]) -> None:
            nonlocal usage
            item_usage = index_one([item], question_id, skip_existing)
            with lock:
                usage += item_usage

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(work, item) for item in numbered]:
                future.result()
        return usage
