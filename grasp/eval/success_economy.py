"""Success economy C_w (Eq. 3): total tokens per difficulty-weighted correct answer.

Each correct answer is weighted by its surprisal w_i = -log2(r_i), where r_i is the
fraction of n zero-shot samples of the backbone that answered question i correctly.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from grasp.datasets import Question
from grasp.eval.metrics import best_exact_match
from grasp.llm import Provider, chat_model
from grasp.prompts import load_prompt

SMOOTHING_N = 10


def success_economy(total_tokens: float, correct: dict[str, float], difficulty: dict[str, float]) -> float:
    floor = 1.0 / (SMOOTHING_N + 1)
    weight = 0.0
    for qid, credit in correct.items():
        if credit > 0.0 and qid in difficulty:
            weight += credit * -math.log2(difficulty[qid] if difficulty[qid] > 0.0 else floor)
    return total_tokens / weight if weight else float("nan")


def bootstrap_interval(
    total_tokens: float,
    correct: dict[str, float],
    difficulty: dict[str, float],
    *,
    samples: int = 10,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% parametric-bootstrap CI: redraw r_i* = Binomial(n, r_i) / n, holding EM and tokens fixed."""
    rng = np.random.default_rng(seed)
    qids = [qid for qid in correct if qid in difficulty]
    credit = np.array([correct[qid] for qid in qids], dtype=np.float64)
    r_hat = np.array([difficulty[qid] for qid in qids], dtype=np.float64)
    draws = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        r_star = rng.binomial(samples, r_hat).astype(np.float64) / samples
        r_star[(r_star == 0.0) & (credit > 0.0)] = 1.0 / (SMOOTHING_N + 1)
        valid = (credit > 0.0) & (r_star > 0.0)
        draws[b] = total_tokens / np.sum(credit[valid] * -np.log2(r_star[valid])) if valid.any() else np.nan
    draws = draws[~np.isnan(draws)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def from_results(results: dict[str, Any], difficulty: dict[str, float]) -> dict[str, float]:
    total = float(results["summary"]["tokens"]["total"]["total_tokens"])
    correct = {r["id"]: float(int(r["exact_match"])) for r in results["questions"]}
    low, high = bootstrap_interval(total, correct, difficulty)
    return {"success_economy": success_economy(total, correct, difficulty), "ci_low": low, "ci_high": high}


class ZeroShotAnswer(BaseModel):
    model_config = ConfigDict(title="_Answer")

    answer: str


def estimate_difficulty(
    questions: list[Question],
    model: str,
    provider: Provider = "gemini",
    base_url: str | None = None,
    *,
    samples: int = 10,
    temperature: float = 1.0,
    with_context: bool = False,
    workers: int = 32,
) -> dict[str, float]:
    """r_i: fraction of `samples` zero-shot answers (temperature 1) that exactly match."""
    timeout = {"request_timeout": 300} if provider == "gemini" else {"timeout": 300}
    llm = chat_model(
        model=model, provider=provider, temperature=temperature, base_url=base_url, thinking_level="minimal", max_retries=3, **timeout
    ).with_structured_output(ZeroShotAnswer)
    system = load_prompt("baseline_no_context")
    correct = {q.id: 0 for q in questions}
    lock = threading.Lock()

    def sample(question: Question) -> None:
        user = context_prompt(question) if with_context and question.passages else f"Question: {question.question}\nAnswer:"
        try:
            raw = llm.invoke([("system", system), ("human", user)]).answer.strip()
        except Exception as error:  # noqa: BLE001
            tqdm.write(f"[{question.id}] {error}")
            return
        parts = raw.split("So the answer is: ")
        prediction = (parts[1].strip() if len(parts) > 1 else raw).removesuffix(".")
        if best_exact_match(prediction, question.answers) == 1.0:
            with lock:
                correct[question.id] += 1

    tasks = [q for _ in range(samples) for q in questions]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(sample, tasks), total=len(tasks), desc="Sampling"))
    return {qid: count / samples for qid, count in correct.items()}


def context_prompt(question: Question) -> str:
    context = "\n\n".join(f"[{p.title}] {p.text}" if p.title else p.text for p in question.passages)
    return f"Use the following passages to answer the question.\n\n{context}\n\nQuestion: {question.question}\nAnswer:"
