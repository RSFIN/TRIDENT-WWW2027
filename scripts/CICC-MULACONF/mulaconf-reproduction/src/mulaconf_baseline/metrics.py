"""Metrics for the five evaluation layers in the paper plan."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Iterable, Sequence

import numpy as np

from .data import LabelSet


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    left, right = set(first), set(second)
    return safe_divide(len(left & right), len(left | right)) if left or right else 1.0


def multilabel_point_metrics(
    truths: Sequence[LabelSet],
    predictions: Sequence[LabelSet],
    ranked_labels: Sequence[Sequence[str]],
) -> dict[str, float]:
    atomic = sorted(
        set().union(*(set(value) for value in truths), *(set(value) for value in predictions))
    )
    f1_values = []
    for label in atomic:
        tp = sum(label in truth and label in pred for truth, pred in zip(truths, predictions))
        fp = sum(label not in truth and label in pred for truth, pred in zip(truths, predictions))
        fn = sum(label in truth and label not in pred for truth, pred in zip(truths, predictions))
        f1_values.append(safe_divide(2 * tp, 2 * tp + fp + fn))
    result = {
        "exact_match": mean(truth == pred for truth, pred in zip(truths, predictions)),
        "intent_set_iou": mean(jaccard(truth, pred) for truth, pred in zip(truths, predictions)),
        "macro_f1": mean(f1_values) if f1_values else 0.0,
    }
    for k in (1, 3, 5):
        result[f"hit_at_{k}"] = mean(
            bool(set(truth) & set(ranking[:k]))
            for truth, ranking in zip(truths, ranked_labels)
        )
    return result


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for truth, score in zip(y_true, scores) if truth]
    negatives = [score for truth, score in zip(y_true, scores) if not truth]
    if not positives or not negatives:
        return math.nan
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return math.nan
    ranked = sorted(zip(scores, y_true), reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, truth) in enumerate(ranked, start=1):
        if truth:
            hits += 1
            total += hits / rank
    return total / positives


def binary_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return safe_divide(2 * tp, 2 * tp + fp + fn)


def threshold_for_recall(y_true: Sequence[int], scores: Sequence[float], target: float) -> float:
    positives = sorted((score for truth, score in zip(y_true, scores) if truth), reverse=True)
    if not positives:
        return math.inf
    return positives[max(1, math.ceil(target * len(positives))) - 1]


def fpr_at_recall(y_true: Sequence[int], scores: Sequence[float], target: float) -> float:
    threshold = threshold_for_recall(y_true, scores, target)
    negatives = [score for truth, score in zip(y_true, scores) if not truth]
    return mean(score >= threshold for score in negatives) if negatives else math.nan


def expected_calibration_error(
    correctness: Sequence[bool], confidences: Sequence[float], bins: int
) -> tuple[float, list[dict[str, float | int]]]:
    rows = []
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidences)
            if lower <= value and (value <= upper if index == bins - 1 else value < upper)
        ]
        if not members:
            rows.append(
                {"lower": lower, "upper": upper, "count": 0, "accuracy": 0.0, "confidence": 0.0}
            )
            continue
        accuracy = mean(correctness[i] for i in members)
        confidence = mean(confidences[i] for i in members)
        error += len(members) / len(confidences) * abs(accuracy - confidence)
        rows.append(
            {"lower": lower, "upper": upper, "count": len(members), "accuracy": accuracy, "confidence": confidence}
        )
    return error, rows


def multiclass_brier(
    probabilities: Sequence[np.ndarray], truth_indices: Sequence[int | None]
) -> float:
    values = []
    for probs, truth in zip(probabilities, truth_indices):
        target = np.zeros_like(probs)
        if truth is not None:
            target[truth] = 1.0
        values.append(float(np.square(probs - target).sum()))
    return mean(values)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def size_distribution(regions: Sequence[Sequence[LabelSet]]) -> dict[str, int]:
    return {str(size): count for size, count in sorted(Counter(map(len, regions)).items())}
