"""Proposition and entity extraction (Figure 3), one passage or a batch per LLM call.

Gemini runs use the raw google-genai SDK, as in the paper; vLLM runs go through LangChain.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from grasp.graph.store import Extraction
from grasp.llm import Backbone, chat_model, google_client, google_retry
from grasp.prompts import load_prompt
from grasp.tokens import TokenCounter, TokenUsage

logger = logging.getLogger(__name__)

ExtractionStrategy = Literal["joint", "batch-joint", "sentence", "batch-sentence"]
EXTRACTION_TEMPERATURE = 0.1


@google_retry
def call_gemini(model: str, prompt: str, config: Any) -> Any:
    return google_client().models.generate_content(model=model, contents=prompt, config=config)


def generate_text(backbone: Backbone, prompt: str) -> tuple[str, TokenUsage]:
    if backbone.provider == "gemini":
        return generate_gemini(backbone.model, prompt)
    counter = TokenCounter()
    llm = chat_model(
        model=backbone.model,
        provider=backbone.provider,
        base_url=backbone.base_url,
        temperature=EXTRACTION_TEMPERATURE,
        thinking_level="minimal",
        max_retries=3,
    )
    response = llm.invoke(prompt, config=counter.config)
    return response.content or "", counter.usage


def generate_gemini(model: str, prompt: str) -> tuple[str, TokenUsage]:
    from google.genai import types

    config: dict[str, Any] = {"temperature": EXTRACTION_TEMPERATURE}
    if model == "gemini-3-flash-preview":
        config["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    response = call_gemini(model, prompt, types.GenerateContentConfig(**config))
    metadata = getattr(response, "usage_metadata", None)
    usage = TokenUsage(
        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
        total_tokens=getattr(metadata, "total_token_count", 0) or 0,
        call_count=1,
    )
    feedback = getattr(response, "prompt_feedback", None)
    if getattr(feedback, "block_reason", None) or not response.candidates:
        logger.warning("Extraction blocked by safety filter; skipping.")
        return "", usage
    return response.text or "", usage


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.strip())
    return [s.strip() for s in parts if s.strip()]


def parse_entities(line: str) -> list | None:
    parts = line.split("|")
    if len(parts) < 3:
        return None
    name, entity_type = parts[0].strip(), parts[1].strip()
    indices = [int(x) for x in parts[2].strip().split() if x.isdecimal()]
    return [name, entity_type, indices] if name and entity_type else None


def parse_joint(text: str) -> tuple[list[str], list[list]]:
    propositions: list[str] = []
    entities: list[list] = []
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("propositions:"):
            section = "propositions"
        elif lower.startswith("entities:"):
            section = "entities"
        elif section == "propositions":
            match = re.match(r"^\[(\d+)\]\s+(.*)", stripped)
            if match:
                propositions.append(match.group(2).strip())
        elif section == "entities" and (entity := parse_entities(stripped)):
            entities.append(entity)
    return propositions, entities


def parse_sentence_entities(text: str) -> list[list]:
    entities: list[list] = []
    in_entities = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("entities:"):
            in_entities = True
        elif in_entities and (entity := parse_entities(stripped)):
            entities.append(entity)
    return entities


def split_passage_sections(text: str) -> dict[int, str]:
    sections: dict[int, str] = {}
    current: int | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*Passage\s*\[(\d+)\]\s*:", line, re.IGNORECASE)
        if match:
            if current is not None:
                sections[current] = "\n".join(lines)
            current, lines = int(match.group(1)), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def link_entities(entities: list[list], num_propositions: int) -> list[list[tuple[str, str]]]:
    linked: list[list[tuple[str, str]]] = [[] for _ in range(num_propositions)]
    for entity in entities:
        if not entity or len(entity) < 3:
            continue
        name, entity_type = str(entity[0]).strip(), str(entity[1]).strip()
        indices = entity[2] if isinstance(entity[2], list) else []
        if not name or not entity_type:
            continue
        for index in indices:
            if 0 <= index < num_propositions and (name, entity_type) not in linked[index]:
                linked[index].append((name, entity_type))
    return linked


def joint_extraction(text: str) -> Extraction:
    propositions, entities = parse_joint(text)
    return Extraction(propositions=propositions, entities=link_entities(entities, len(propositions)))


class Extractor:
    """Turns (title, text) passages into propositions with linked, typed entities."""

    def __init__(self, strategy: ExtractionStrategy, backbone: Backbone, batch_size: int = 10) -> None:
        self.strategy = strategy
        self.backbone = backbone
        self.batch_size = batch_size if strategy.startswith("batch") else 1

    def extract(self, passages: list[tuple[str, str]]) -> tuple[list[Extraction], TokenUsage]:
        if self.strategy == "batch-joint":
            return self.extract_batch_joint(passages)
        if self.strategy == "batch-sentence":
            return self.extract_batch_sentence(passages)
        single = self.extract_joint if self.strategy == "joint" else self.extract_sentence
        extractions, usage = [], TokenUsage()
        for title, text in passages:
            extraction, call_usage = single(title, text)
            extractions.append(extraction)
            usage += call_usage
        return extractions, usage

    def extract_joint(self, title: str, text: str) -> tuple[Extraction, TokenUsage]:
        response, usage = generate_text(self.backbone, load_prompt("joint_extraction", title=title, passage=text))
        if not response:
            logger.warning("Empty extraction for %r", title[:80])
            return Extraction(propositions=[]), usage
        return joint_extraction(response), usage

    def extract_sentence(self, title: str, text: str) -> tuple[Extraction, TokenUsage]:
        sentences = split_sentences(text)
        if not sentences:
            return Extraction(propositions=[]), TokenUsage()
        numbered = "\n".join(f"[{i}]: {s}" for i, s in enumerate(sentences))
        response, usage = generate_text(self.backbone, load_prompt("sentence_entity_extraction", title=title, numbered_sentences=numbered))
        if not response:
            return Extraction(propositions=sentences, entities=[[] for _ in sentences]), usage
        return Extraction(propositions=sentences, entities=link_entities(parse_sentence_entities(response), len(sentences))), usage

    def extract_batch_joint(self, passages: list[tuple[str, str]]) -> tuple[list[Extraction], TokenUsage]:
        block = "\n\n".join(f"Passage [{i}]:\nDocument Title: {title}\nContent: {text}" for i, (title, text) in enumerate(passages))
        response, usage = generate_text(self.backbone, load_prompt("batch_joint_extraction", passages_block=block))
        sections = split_passage_sections(response) if response else {}
        extractions = []
        for i in range(len(passages)):
            if not sections.get(i, "").strip():
                logger.warning("Batch extraction missing passage %d/%d", i, len(passages))
                extractions.append(Extraction(propositions=[]))
            else:
                extractions.append(joint_extraction(sections[i]))
        return extractions, usage

    def extract_batch_sentence(self, passages: list[tuple[str, str]]) -> tuple[list[Extraction], TokenUsage]:
        per_passage = [split_sentences(text) for _, text in passages]
        block = "\n\n".join(
            f"Passage [{i}]:\nDocument Title: {title}\nSentences:\n" + "\n".join(f"[{j}]: {s}" for j, s in enumerate(sentences))
            for i, ((title, _), sentences) in enumerate(zip(passages, per_passage))
        )
        response, usage = generate_text(self.backbone, load_prompt("batch_sentence_extraction", passages_block=block))
        sections = split_passage_sections(response) if response else {}
        extractions = []
        for i, sentences in enumerate(per_passage):
            section = sections.get(i, "")
            if response and not section.strip():
                logger.warning("Batch extraction missing passage %d/%d", i, len(passages))
            extractions.append(Extraction(propositions=sentences, entities=link_entities(parse_sentence_entities(section), len(sentences))))
        return extractions, usage
