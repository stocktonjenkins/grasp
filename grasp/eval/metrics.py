"""Exact match and token F1 (SQuAD-style normalization, as in HippoRAG2 and GeAR)."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


def f1(prediction: str, truth: str) -> float:
    predicted, gold = normalize_answer(prediction).split(), normalize_answer(truth).split()
    if not predicted or not gold:
        return float(predicted == gold)
    common = sum((Counter(predicted) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, truths: list[str]) -> float:
    return max((exact_match(prediction, t) for t in truths), default=0.0)


def best_f1(prediction: str, truths: list[str]) -> float:
    return max((f1(prediction, t) for t in truths), default=0.0)
