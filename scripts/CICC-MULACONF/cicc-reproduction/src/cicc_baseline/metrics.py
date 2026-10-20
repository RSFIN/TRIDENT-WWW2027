"""Dependency-free metrics for the five evaluation layers in the paper plan."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Mapping, Sequence


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("non-empty y_true and y_pred must have equal length")
    return mean(true == predicted for true, predicted in zip(y_true, y_pred))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("non-empty y_true and y_pred must have equal length")
    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        scores.append(safe_divide(2.0 * precision * recall, precision + recall))
    return mean(scores)


def binary_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_positive = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return safe_divide(2.0 * precision * recall, precision + recall)


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("non-empty labels and scores must have equal length")
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return math.nan
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def threshold_for_recall(y_true: Sequence[int], scores: Sequence[float], target_recall: float) -> float:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must lie in (0, 1]")
    positive_scores = sorted((score for label, score in zip(y_true, scores) if label == 1), reverse=True)
    if not positive_scores:
        raise ValueError("at least one positive example is required")
    rank = max(1, math.ceil(target_recall * len(positive_scores)))
    return positive_scores[rank - 1]


def fpr_at_recall(y_true: Sequence[int], scores: Sequence[float], target_recall: float = 0.95) -> float:
    threshold = threshold_for_recall(y_true, scores, target_recall)
    negatives = [score for label, score in zip(y_true, scores) if label == 0]
    return safe_divide(sum(score >= threshold for score in negatives), len(negatives))


def expected_calibration_error(
    correct: Sequence[bool], confidences: Sequence[float], bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    if len(correct) != len(confidences) or not correct:
        raise ValueError("non-empty correctness and confidence arrays must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    rows: list[dict[str, float]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            rows.append({"lower": lower, "upper": upper, "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        bin_accuracy = mean(correct[position] for position in members)
        bin_confidence = mean(confidences[position] for position in members)
        ece += len(members) / len(correct) * abs(bin_accuracy - bin_confidence)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
            }
        )
    return ece, rows


def multiclass_brier(probability_rows: Sequence[Mapping[str, float]], true_labels: Sequence[str]) -> float:
    if len(probability_rows) != len(true_labels) or not probability_rows:
        raise ValueError("non-empty probabilities and labels must have equal length")
    losses = []
    for row, true_label in zip(probability_rows, true_labels):
        losses.append(sum((probability - float(label == true_label)) ** 2 for label, probability in row.items()))
    return mean(losses)


def empirical_coverage(candidate_sets: Sequence[Sequence[str]], true_labels: Sequence[str]) -> float:
    if len(candidate_sets) != len(true_labels) or not candidate_sets:
        raise ValueError("non-empty candidate sets and labels must have equal length")
    return mean(label in candidates for candidates, label in zip(candidate_sets, true_labels))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def set_size_distribution(candidate_sets: Sequence[Sequence[str]]) -> dict[str, int]:
    counts = Counter(len(candidates) for candidates in candidate_sets)
    return {str(size): counts[size] for size in sorted(counts)}

