"""Model access for the two paper backbones: Gemini (Google AI) and Qwen3 served by vLLM.

This module is the only place a Gemini client is configured. `google_client()` is the
shared google-genai SDK client (embeddings and extraction). `chat_model()` builds the
LangChain chat model used by the agent, with the same API key. LangChain creates its own
SDK client internally and closes it on garbage collection, so it cannot share ours.
"""

from __future__ import annotations

import functools
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Literal

from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from google import genai

Provider = Literal["gemini", "vllm"]

TRANSIENT_RETRY: dict[str, Any] = dict(
    retry_if_exception_type=(google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable),
    wait_exponential_jitter=True,
    stop_after_attempt=5,
)

GOOGLE_TRANSIENT = (
    google_exceptions.ServiceUnavailable,
    google_exceptions.ResourceExhausted,
    google_exceptions.InternalServerError,
    google_exceptions.DeadlineExceeded,
    genai_errors.ServerError,
    genai_errors.ClientError,
)

google_retry = retry(
    retry=retry_if_exception_type(GOOGLE_TRANSIENT),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


def google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to call Gemini.")
    return api_key


@functools.cache
def google_client() -> genai.Client:
    """The google-genai client, created on first use and shared by every caller and thread."""
    from google import genai

    return genai.Client(api_key=google_api_key())


def supports_thinking(model: str, provider: str) -> bool:
    if provider == "vllm":
        return "qwen3" in model.lower()
    match = re.match(r"gemini-(\d+)", model)
    return match is not None and int(match.group(1)) >= 3


def chat_model(
    model: str,
    provider: Provider = "gemini",
    temperature: float = 0.0,
    *,
    max_retries: int | None = None,
    thinking_level: str | None = None,
    base_url: str | None = None,
    **extra: Any,
) -> BaseChatModel:
    thinking = thinking_level is not None and supports_thinking(model, provider)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature, "google_api_key": google_api_key(), **extra}
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if thinking:
            kwargs["thinking_level"] = thinking_level
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": model,
            "temperature": temperature,
            "base_url": base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
            **extra,
        }
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if thinking:
            enable = thinking_level not in ("minimal", "none")
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable}}
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown provider {provider!r}; expected 'gemini' or 'vllm'.")


ChatBuilder = Callable[..., BaseChatModel]


class Backbone(BaseModel):
    """The model every GRASP call runs on, at a caller-chosen temperature."""

    model_config = ConfigDict(frozen=True)

    model: str = "gemini-3-flash-preview"
    provider: Provider = "gemini"
    base_url: str | None = None
    build: ChatBuilder = chat_model

    def chat(self, temperature: float) -> BaseChatModel:
        return self.build(
            model=self.model,
            provider=self.provider,
            temperature=temperature,
            max_retries=3,
            thinking_level="minimal",
            base_url=self.base_url,
        )
