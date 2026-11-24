"""Flat dense passage index (parquet) used by the RAG, Q-Dec and Self-Ask baselines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from grasp.embeddings import Embedder

if TYPE_CHECKING:
    import tiktoken

CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50


def chunk_passage(text: str, encoder: tiktoken.Encoding) -> list[str]:
    tokens = encoder.encode(text)
    if len(tokens) <= CHUNK_TOKENS:
        return [text]
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - CHUNK_OVERLAP
    return chunks


def build_dense_index(scoped_passages: list[tuple[str, str, str]], embedder: Embedder, path: Path) -> int:
    """Index (scope_id, title, text) passages; scope is the question id for per-question retrieval."""
    import polars as pl
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    rows: dict[str, dict] = {}
    position: dict[str, int] = {}
    for scope, title, text in scoped_passages:
        text = text.strip()
        if not text:
            continue
        for chunk in chunk_passage(text, encoder):
            index = position.get(scope, 0)
            position[scope] = index + 1
            rows.setdefault(f"{scope}/{index}", {"id": f"{scope}/{index}", "question_id": scope, "title": title, "text": chunk})

    parquet = Path(path) / "passages.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    existing = set(pl.read_parquet(parquet, columns=["id"])["id"].to_list()) if parquet.exists() else set()
    new = [row for row_id, row in rows.items() if row_id not in existing]
    if not new:
        return 0
    vectors = embedder.documents([row["text"] for row in new])
    frame = pl.DataFrame(
        {
            "id": [r["id"] for r in new],
            "question_id": [r["question_id"] for r in new],
            "passage_id": [hashlib.md5(f"{r['title']}||{r['text'][:200]}".encode()).hexdigest()[:16] for r in new],
            "title": [r["title"] for r in new],
            "text": [r["text"] for r in new],
            "embedding": vectors,
        }
    )
    (pl.concat([pl.read_parquet(parquet), frame]) if parquet.exists() else frame).write_parquet(parquet)
    return len(new)


class DenseIndex(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    texts: list[str]
    embeddings: np.ndarray
    rows_by_question: dict[str, list[int]]

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        import polars as pl

        frame = pl.read_parquet(Path(path) / "passages.parquet")
        embeddings = np.stack(frame["embedding"].to_list()).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        rows: dict[str, list[int]] = {}
        for i, question_id in enumerate(frame["question_id"].to_list()):
            rows.setdefault(question_id, []).append(i)
        return cls(texts=frame["text"].to_list(), embeddings=embeddings / np.where(norms > 0, norms, 1.0), rows_by_question=rows)

    def retrieve(self, query: np.ndarray, question_id: str, top_k: int, global_retrieval: bool = False) -> list[str]:
        if global_retrieval:
            order = np.argsort(self.embeddings @ query)[::-1][:top_k]
            return [self.texts[i] for i in order]
        rows = self.rows_by_question.get(question_id)
        if not rows:
            return []
        order = np.argsort(self.embeddings[rows] @ query)[::-1][:top_k]
        return [self.texts[rows[i]] for i in order]
