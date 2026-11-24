"""Benchmarks: MuSiQue, 2Wiki and HotpotQA (HippoRAG2 retrieval splits) and their LongBench variants."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

LONGBENCH_CONFIGS = {"longbench_musique": "musique", "longbench_2wiki": "2wikimqa", "longbench_hotpot": "hotpotqa"}


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    text: str
    is_supporting: bool = False


class Question(BaseModel):
    id: str
    question: str
    answers: list[str]
    passages: list[Passage] = Field(default_factory=list)
    decomposition: list[dict] = Field(default_factory=list)

    def indexable_passages(self) -> list[tuple[str, str]]:
        return [(p.title, p.text) for p in self.passages if p.text.strip()]


class Chunking(BaseModel):
    """LongBench passage chunking: greedy paragraph packing up to `max_tokens` (tiktoken)."""

    max_tokens: int = 500
    min_tokens: int = 0
    encoding: str = "gpt-4"


def load_questions(dataset: str, data: Path | None, chunking: Chunking | None = None) -> list[Question]:
    if dataset in LONGBENCH_CONFIGS:
        return load_longbench(LONGBENCH_CONFIGS[dataset], chunking, data)
    if data is None:
        raise ValueError(f"Dataset {dataset!r} needs a `data` directory containing qa.json.")
    records = json.loads((Path(data) / "qa.json").read_text(encoding="utf-8"))
    if dataset == "musique":
        return [musique_question(r) for r in records if r.get("id")]
    if dataset in ("2wiki", "hotpot"):
        return [wikimultihop_question(r) for r in records if r.get("_id")]
    raise ValueError(f"Unknown dataset {dataset!r}.")


def load_corpus(data: Path) -> list[tuple[str, str]]:
    entries = json.loads((Path(data) / "corpus.json").read_text(encoding="utf-8"))
    return [(entry.get("title", ""), entry.get("text", "").strip()) for entry in entries if entry.get("text", "").strip()]


def musique_question(record: dict) -> Question:
    return Question(
        id=str(record["id"]),
        question=record.get("question", ""),
        answers=[record.get("answer", "")] + list(record.get("answer_aliases") or []),
        passages=[
            Passage(
                title=p.get("title", ""), text=p.get("paragraph_text", "") or p.get("text", ""), is_supporting=bool(p.get("is_supporting"))
            )
            for p in record.get("paragraphs", [])
        ],
        decomposition=record.get("question_decomposition", []),
    )


def wikimultihop_question(record: dict) -> Question:
    supporting = {title for title, _ in record.get("supporting_facts", [])}
    return Question(
        id=str(record["_id"]),
        question=record.get("question", ""),
        answers=[record.get("answer", "")],
        passages=[
            Passage(title=title, text=" ".join(s.strip() for s in sentences if s.strip()), is_supporting=title in supporting)
            for title, sentences in record.get("context", [])
        ],
    )


def load_longbench(config: str, chunking: Chunking | None, cache_dir: Path | None) -> list[Question]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "test", **({"cache_dir": str(cache_dir)} if cache_dir else {})}
    try:
        records = load_dataset("THUDM/LongBench", config, **kwargs)
    except Exception as error:  # noqa: BLE001
        logger.warning("THUDM/LongBench unavailable (%s); using the yanbingzheng/LongBench mirror.", error)
        records = load_dataset("yanbingzheng/LongBench", config, **kwargs)

    count = token_counter(chunking.encoding) if chunking else None
    questions = []
    for i, record in enumerate(records):
        passages = split_longbench_context(record.get("context", ""))
        if chunking and count:
            passages = [
                Passage(title=p.title, text=part)
                for p in passages
                for part in chunk_by_paragraphs(p.text.strip(), chunking.max_tokens, chunking.min_tokens, count)
            ]
        answers = record.get("answers") or []
        questions.append(
            Question(
                id=str(record.get("_id") or record.get("id") or f"{config}_{i}"),
                question=record.get("input", ""),
                answers=[answers] if isinstance(answers, str) else list(answers),
                passages=passages,
            )
        )
    return questions


def split_longbench_context(context: str) -> list[Passage]:
    passages = []
    for part in re.split(r"(?:^|\n)Passage \d+:\n", context, flags=re.MULTILINE):
        title, _, text = part.strip().partition("\n")
        if text.strip():
            passages.append(Passage(title=title.strip(), text=text.strip()))
    return passages


def token_counter(encoding: str) -> Callable[[str], int]:
    import tiktoken

    encoder = tiktoken.encoding_for_model(encoding)
    return lambda text: len(encoder.encode(text))


def chunk_by_paragraphs(text: str, max_tokens: int, min_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Pack paragraphs into chunks of at most `max_tokens`.

    Returns the passage unsplit if any chunk would fall under `min_tokens`.
    """
    if count(text) <= max_tokens:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return chunk_by_tokens(text, max_tokens, count)

    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = current + [paragraph]
        if not current or count("\n\n".join(candidate)) <= max_tokens:
            current = candidate
            continue
        chunks.append("\n\n".join(current))
        if count(paragraph) > max_tokens:
            chunks.extend(chunk_by_tokens(paragraph, max_tokens, count))
            current = []
        else:
            current = [paragraph]
    if current:
        chunks.append("\n\n".join(current))

    if min_tokens > 0 and any(count(chunk) < min_tokens for chunk in chunks):
        return [text]
    return chunks


def chunk_by_tokens(text: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    if count(text) <= max_tokens:
        return [text]
    head = longest_prefix(text, max_tokens, count)
    if len(head) >= len(text) or count(text[len(head) :]) <= 0:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = longest_prefix(remaining, max_tokens, count)
        if not chunk:
            break
        chunks.append(chunk)
        if len(chunk) >= len(remaining):
            break
        remaining = remaining[len(chunk) :]
    return chunks or [text]


def longest_prefix(text: str, max_tokens: int, count: Callable[[str], int]) -> str:
    if not text or count(text) <= max_tokens:
        return text
    low, high, best = 1, len(text), ""
    while low <= high:
        mid = (low + high) // 2
        if count(text[:mid]) <= max_tokens:
            best, low = text[:mid], mid + 1
        else:
            high = mid - 1
    return best or text[:1]
