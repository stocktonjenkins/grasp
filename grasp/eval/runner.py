"""Runs a method over a benchmark, checkpointing every answer to a results JSON file."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from grasp.datasets import Question
from grasp.eval.metrics import best_exact_match, best_f1
from grasp.methods import Method, Prediction
from grasp.tokens import TokenUsage

logger = logging.getLogger(__name__)


def read_results(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def summarize(records: list[dict], index_usage: TokenUsage) -> dict[str, Any]:
    n = len(records) or 1
    inference = sum((TokenUsage.from_dict(r["token_usage"]) for r in records), TokenUsage())
    construction = sum((TokenUsage.from_dict(r.get("construction_token_usage")) for r in records), index_usage)
    summary: dict[str, Any] = {
        "questions": len(records),
        "errors": sum(1 for r in records if r.get("error")),
        "exact_match": sum(r["exact_match"] for r in records) / n,
        "f1": sum(r["f1"] for r in records) / n,
        "tokens": {
            "construction": construction.to_dict(),
            "inference": inference.to_dict(),
            "total": (construction + inference).to_dict(),
        },
    }
    judged = [r for r in records if "llm_judge" in r]
    if judged:
        summary["llm_judge"] = sum(r["llm_judge"] for r in judged) / len(judged)
    return summary


def answer_question(method: Method, question: Question) -> dict[str, Any]:
    record: dict[str, Any] = {"id": question.id, "question": question.question, "ground_truths": question.answers}
    start = time.time()
    try:
        prediction = method.answer(question)
    except Exception as error:  # noqa: BLE001
        prediction = Prediction(answer="", usage=TokenUsage(), error=str(error))
    if prediction.error:
        logger.warning("[%s] failed: %s", question.id, prediction.error)
    return {
        **record,
        "prediction": prediction.answer,
        "exact_match": best_exact_match(prediction.answer, question.answers),
        "f1": best_f1(prediction.answer, question.answers),
        "token_usage": prediction.usage.to_dict(),
        "construction_token_usage": prediction.construction_usage.to_dict(),
        "construction_seconds": prediction.construction_seconds,
        "seconds": time.time() - start,
        "rational_plan": prediction.rational_plan,
        "rationale": prediction.rationale,
        "trace": prediction.trace,
        **({"error": prediction.error} if prediction.error else {}),
    }


def carry_failed_usage(record: dict[str, Any], failed: dict[str, dict]) -> dict[str, Any]:
    """Tokens spent on an earlier failed attempt still count toward the run's total."""
    if record["id"] not in failed:
        return record
    earlier = failed[record["id"]]
    for key in ("token_usage", "construction_token_usage"):
        record[key] = (TokenUsage.from_dict(record.get(key)) + TokenUsage.from_dict(earlier.get(key))).to_dict()
    return record


def as_completed_or_cancel[T](pool: ThreadPoolExecutor, futures: list[Future[T]]) -> Iterator[Future[T]]:
    """`as_completed`, but an interrupt (Ctrl-C, a job's time budget) drops the queued work instead of finishing it."""
    try:
        yield from as_completed(futures)
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise


def evaluate(
    method: Method,
    questions: list[Question],
    output: Path,
    *,
    config: dict[str, Any],
    parallel: int = 1,
    resume: bool = False,
    index_usage: TokenUsage | None = None,
) -> dict[str, Any]:
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; pass --resume to continue it or --output to write elsewhere.")
    previous = {r["id"]: r for r in read_results(output)["questions"]} if output.exists() else {}
    done = {qid: r for qid, r in previous.items() if not r.get("error")}
    failed = {qid: r for qid, r in previous.items() if r.get("error")}
    order = {q.id: i for i, q in enumerate(questions)}
    pending = [q for q in questions if q.id not in done]
    index_usage = index_usage or TokenUsage()
    lock = threading.Lock()

    def checkpoint() -> dict[str, Any]:
        records = sorted(done.values(), key=lambda r: order.get(r["id"], len(order)))
        results = {"config": config, "summary": summarize(records, index_usage), "questions": records}
        write_results(output, results)
        return results

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool, tqdm(total=len(pending), desc="Answering") as bar:
        for future in as_completed_or_cancel(pool, [pool.submit(answer_question, method, q) for q in pending]):
            record = carry_failed_usage(future.result(), failed)
            with lock:
                done[record["id"]] = record
                results = checkpoint()
            bar.update(1)
            bar.set_postfix(EM=f"{results['summary']['exact_match']:.3f}", F1=f"{results['summary']['f1']:.3f}")
    return checkpoint()
