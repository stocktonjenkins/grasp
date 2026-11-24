"""Passage recall on MuSiQue (Table 2).

`decomposed`: each gold sub-question is rewritten by GRASP's rewriter (with gold prior
answers), retrieves its top-k passages, and scores a hit if its gold passage is
among them. `single`: one retrieval with the full multi-hop question, recall over
all gold passages.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from grasp.llm import chat_model
from grasp.prompts import load_prompt

logger = logging.getLogger(__name__)

Retrieve = Callable[[str, list[str]], list[dict]]


class RewriteResult(BaseModel):
    search_statement: str = Field(
        description=("Short declarative statement optimized for dense retrieval, following the rules in the prompt.")
    )
    keywords: list[str] = Field(
        description=(
            """
            Keywords for fulltext search, extracted from the search statement.
            **Keyword extraction rules:**
                - **Prioritize Proper Nouns:** Extract specific people, places, organizations, or distinct concepts. Avoid extracting common nouns (e.g., "movie," "company," "budget") unless they are part of an entity's name.
                - **Group Compound Entities:** Do not split full names or multi-word entities. Good: `["Ohio State", "Michigan"]`. Bad: `["Ohio", "State", "Michigan"]`.
                - **Remove Noise:** Do not include question words (who, what, when, where) or verbs (is, was, played, directed).
                - If the subquestion is "What awards did Christopher Nolan win?", the keywords should be `["Christopher Nolan"]`. NOT `["awards", "Christopher Nolan"]`.
            """
        )
    )


def rewrite_gold_subquestions(items: list[dict], model: str, *, n: int = 100, seed: int = 42, workers: int = 1) -> list[dict]:
    items = [item for item in items if item.get("answerable", True)]
    if 0 < n < len(items):
        items = random.Random(seed).sample(items, n)
    llm = chat_model(model, "gemini", 0, max_retries=3, thinking_level="minimal").with_structured_output(RewriteResult)

    def rewrite(item: dict) -> dict:
        steps = item.get("question_decomposition", [])
        plan = "\n".join(f"{i + 1}. {step['question']}" for i, step in enumerate(steps))
        rewritten = []
        for i, step in enumerate(steps):
            history = "\n".join(f"Step {j + 1} Answer: {steps[j].get('answer', '')}" for j in range(i))
            prompt = load_prompt(
                "rewriter",
                original_question=item["question"],
                rational_plan=plan,
                context_history=history,
                current_sub_question=step["question"],
            )
            try:
                result = llm.invoke(prompt)
                statement, keywords = result.search_statement, result.keywords
            except Exception as error:  # noqa: BLE001
                logger.warning("Rewrite failed for %s step %d: %s", item["id"], i + 1, error)
                statement, keywords = step["question"], []
            rewritten.append(
                {
                    "step": i + 1,
                    "raw_subquestion": step["question"],
                    "gold_answer": step.get("answer", ""),
                    "paragraph_support_idx": step.get("paragraph_support_idx"),
                    "search_statement": statement,
                    "keywords": keywords,
                }
            )
        return {**item, "rewritten_subquestions": rewritten}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(rewrite, items))


def question_recall(item: dict, retrieve: Retrieve, mode: Literal["decomposed", "single"], k: int = 5) -> dict[str, Any] | None:
    paragraphs = item.get("paragraphs", [])
    steps = item.get("rewritten_subquestions", [])
    if not steps:
        return None
    if mode == "single":
        gold: list[str] = []
        keywords: list[str] = []
        for step in steps:
            index = step.get("paragraph_support_idx")
            if index is not None and index < len(paragraphs) and paragraphs[index]["paragraph_text"] not in gold:
                gold.append(paragraphs[index]["paragraph_text"])
            keywords += [kw for kw in step.get("keywords") or [] if kw not in keywords]
        top = {p["text"] for p in retrieve(item.get("question", ""), keywords)[:k]}
        return {"item_id": item["id"], "hop_count": len(steps), "n_gold": len(gold), "recall": len(set(gold) & top) / len(set(gold))}

    hits = []
    for step in steps:
        index = step.get("paragraph_support_idx")
        if not step.get("search_statement", "") or index is None:
            continue
        if index >= len(paragraphs):
            logger.warning("paragraph_support_idx %d out of range for %s", index, item["id"])
            continue
        top = {p["text"] for p in retrieve(step["search_statement"], step.get("keywords") or [])[:k]}
        hits.append(1.0 if paragraphs[index]["paragraph_text"] in top else 0.0)
    return {"item_id": item["id"], "hop_count": len(steps), "n_steps": len(hits), "recall": sum(hits) / len(hits)}
