from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            call_count=self.call_count + other.call_count,
        )

    def to_dict(self) -> dict[str, int]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TokenUsage:
        data = data or {}
        return cls(**{name: int(data.get(name, 0)) for name in cls.model_fields})


def usage_from(response: LLMResult) -> TokenUsage:
    try:
        metadata = response.generations[0][0].message.usage_metadata or {}
    except (AttributeError, IndexError, TypeError):
        metadata = {}
    return TokenUsage(
        input_tokens=metadata.get("input_tokens", 0),
        output_tokens=metadata.get("output_tokens", 0),
        total_tokens=metadata.get("total_tokens", 0),
        call_count=1,
    )


class TokenCounter(BaseCallbackHandler):
    """Thread-safe accumulator of LangChain LLM token usage."""

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = usage_from(response)
        with self._lock:
            self.usage = self.usage + usage

    def add(self, usage: TokenUsage) -> None:
        with self._lock:
            self.usage = self.usage + usage

    @property
    def config(self) -> RunnableConfig:
        return {"callbacks": [self]}
