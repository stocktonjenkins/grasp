"""Two-rater LLM judge from GraphReader/ReadAgent (strict LR-1, lenient LR-2), used in Table 6."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from grasp.eval.metrics import best_exact_match, best_f1
from grasp.llm import chat_model
from grasp.tokens import TokenCounter


class LR1Judgment(BaseModel):
    """LR-1: strict rater — Yes / No only."""

    verdict: Literal["Yes", "No"]


class LR2Judgment(BaseModel):
    """LR-2: lenient rater — allows partial credit."""

    verdict: Literal["Yes", "Yes, partially", "No"]


def format_truths(ground_truths: list[str]) -> str:
    truths = [t for t in ground_truths if t]
    if not truths:
        return "(none provided)"
    if len(truths) == 1:
        return truths[0]
    return "(any one is acceptable)\n" + "\n".join(f"- {t}" for t in truths)


def judge_prompt(question: str, prediction: str, ground_truths: list[str]) -> str:
    return (
        "After reading some text, John was given the following question about the text:\n"
        f"{question}\n"
        "John's answer to the question was:\n"
        f"{prediction}\n"
        "The ground truth answer was:\n"
        f"{format_truths(ground_truths)}\n"
        "Does John's answer agree with the ground truth answer?\n"
    )


def lr1_prompt(question: str, prediction: str, ground_truths: list[str]) -> str:
    return judge_prompt(question, prediction, ground_truths) + 'Please answer "Yes" or "No".'


def lr2_prompt(question: str, prediction: str, ground_truths: list[str]) -> str:
    return judge_prompt(question, prediction, ground_truths) + (
        'Please answer "Yes", "Yes, partially", or "No". '
        'If John\'s response has any overlap with the ground truth answer, answer "Yes, partially". '
        'If John\'s response contains the ground truth answer, answer "Yes". '
        'If John\'s response is more specific than the ground truth answer, answer "Yes".'
    )


class Judge:
    """Correct if either rater says "Yes"; partial if LR-1 says "No" and LR-2 "Yes, partially"."""

    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.1) -> None:
        self.model = model
        self.temperature = temperature
        self.counter = TokenCounter()

    def judge(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def rater(schema: type[BaseModel]) -> Runnable:
            return chat_model(self.model, "gemini", self.temperature, thinking_level="minimal").with_structured_output(schema)

        truths = [
            [r["ground_truths"]] if isinstance(r.get("ground_truths"), str) else [str(t) for t in r.get("ground_truths") or []]
            for r in records
        ]
        lr1 = rater(LR1Judgment).batch(
            [[("user", lr1_prompt(str(r.get("question", "")), str(r.get("prediction", "")), t))] for r, t in zip(records, truths)],
            config=self.counter.config,
        )
        lr2 = rater(LR2Judgment).batch(
            [[("user", lr2_prompt(str(r.get("question", "")), str(r.get("prediction", "")), t))] for r, t in zip(records, truths)],
            config=self.counter.config,
        )
        judged = []
        for record, gts, strict, lenient in zip(records, truths, lr1, lr2):
            strict_verdict = strict.verdict if strict is not None else "No"
            lenient_verdict = lenient.verdict if lenient is not None else "No"
            correct = int(strict_verdict == "Yes" or lenient_verdict == "Yes")
            prediction = str(record.get("prediction", ""))
            judged.append(
                {
                    **record,
                    "lr1_verdict": strict_verdict,
                    "lr2_verdict": lenient_verdict,
                    "llm_judge": correct,
                    "llm_judge_partial": int(not correct and lenient_verdict == "Yes, partially"),
                    "exact_match": best_exact_match(prediction, gts),
                    "f1": best_f1(prediction, gts),
                }
            )
        return judged
