"""Text embeddings: gemini-embedding-001 (Gemini runs) or a local BGE model (Qwen runs)."""

from __future__ import annotations

import threading
from typing import Any, Literal

from grasp.llm import google_client, google_retry

Task = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        provider: Literal["gemini", "huggingface"] = "gemini",
        batch_size: int = 100,
    ) -> None:
        self.model = model
        self.provider = provider
        self.batch_size = batch_size
        self._local_model: Any = None
        self._lock = threading.Lock()

    def query(self, text: str) -> list[float]:
        return self.embed([text], "RETRIEVAL_QUERY")[0]

    def documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, "RETRIEVAL_DOCUMENT")

    def embed(self, texts: list[str], task: Task) -> list[list[float]]:
        if self.provider == "gemini":
            return self.embed_gemini(texts, task)
        return self.embed_local(texts, task)

    @google_retry
    def embed_gemini(self, texts: list[str], task: Task) -> list[list[float]]:
        from google.genai import types

        client = google_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = client.models.embed_content(
                model=self.model,
                contents=texts[start : start + self.batch_size],
                config=types.EmbedContentConfig(task_type=task),
            )
            vectors.extend(list(embedding.values) for embedding in response.embeddings)
        return vectors

    def embed_local(self, texts: list[str], task: Task) -> list[list[float]]:
        if task == "RETRIEVAL_QUERY" and "bge" in self.model.lower():
            texts = [BGE_QUERY_INSTRUCTION + text for text in texts]
        encoded = self.sentence_transformer().encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return encoded.tolist()

    def sentence_transformer(self) -> Any:
        with self._lock:
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.model)
        return self._local_model
