"""WWW 2027 unified metric protocol v1.0."""

from __future__ import annotations

from statistics import mean
from typing import Sequence


TABLE_COLUMNS = (
    ("Macro-F1", "macro_f1"),
    ("Set Exact", "set_exact"),
    ("Set IoU", "set_iou"),
    ("Hit@1", "hit_at_1"),
    ("Hit@3", "hit_at_3"),
    ("Cov(J≥0.5)", "coverage_jaccard"),
    ("Cov(含)", "coverage_intersection"),
    ("Avg \\|P\\|", "avg_set_size"),
    ("ECE", "ece_15_bin"),
    ("误拒率", "false_reject_rate"),
    ("OOS拒识率", "oos_reject_rate"),
    ("澄清率", "clarification_rate"),
)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_f1(true: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(true) | set(predicted))
    values = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, predicted))
        fp = sum(a != label and b == label for a, b in zip(true, predicted))
        fn = sum(a == label and b != label for a, b in zip(true, predicted))
        values.append(_safe_div(2 * tp, 2 * tp + fp + fn))
    return mean(values) if values else 0.0


def _ece15(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("ECE requires paired, non-empty inputs")
    if any(value < 0.0 or value > 1.0 for value in confidences):
        raise ValueError("top-1 confidence must be in [0, 1]")
    error = 0.0
    for bin_index in range(15):
        members = [
            index for index, value in enumerate(confidences)
            if min(int(value * 15), 14) == bin_index
        ]
        if members:
            bin_confidence = mean(confidences[index] for index in members)
            bin_accuracy = mean(correct[index] for index in members)
            error += len(members) / len(confidences) * abs(bin_confidence - bin_accuracy)
    return error


def compute_unified_v1(
    *,
    gold_sets: Sequence[Sequence[str]],
    final_sets: Sequence[Sequence[str]],
    rankings: Sequence[Sequence[str]],
    confidences: Sequence[float],
    states: Sequence[str],
    oos_states: Sequence[str],
) -> dict[str, float]:
    size = len(gold_sets)
    if not size or any(len(values) != size for values in (final_sets, rankings, confidences, states)):
        raise ValueError("unified v1.0 inputs must be paired and non-empty")
    for predicted, ranking, state in zip(final_sets, rankings, states):
        if len(predicted) != len(set(predicted)) or set(predicted) != set(ranking):
            raise ValueError("ranking R must contain exactly the unique members of final set P")
        if (state == "reject") != (len(predicted) == 0):
            raise ValueError("unified v1.0 requires reject iff P is empty")
        if state not in {"reject", "release", "clarify"}:
            raise ValueError(f"unknown decision state: {state}")

    main_true = [gold[0] for gold in gold_sets]
    main_predicted = [ranking[0] if ranking else "" for ranking in rankings]
    main_correct = [truth == predicted for truth, predicted in zip(main_true, main_predicted)]
    exact: list[bool] = []
    ious: list[float] = []
    hit1: list[bool] = []
    hit3: list[bool] = []
    contains: list[bool] = []
    for gold, predicted, ranking, state in zip(gold_sets, final_sets, rankings, states):
        gold_set, predicted_set = set(gold), set(predicted)
        rejected = state == "reject"
        exact.append(not rejected and predicted_set == gold_set)
        ious.append(0.0 if rejected else _safe_div(len(predicted_set & gold_set), len(predicted_set | gold_set)))
        hit1.append(not rejected and gold_set.issubset(ranking[:1]))
        hit3.append(not rejected and gold_set.issubset(ranking[:3]))
        contains.append(not rejected and bool(predicted_set & gold_set))

    return {
        "macro_f1": _macro_f1(main_true, main_predicted),
        "set_exact": mean(exact),
        "set_iou": mean(ious),
        "hit_at_1": mean(hit1),
        "hit_at_3": mean(hit3),
        "coverage_jaccard": mean(value >= 0.5 for value in ious),
        "coverage_intersection": mean(contains),
        "avg_set_size": mean(len(predicted) for predicted in final_sets),
        "ece_15_bin": _ece15(confidences, main_correct),
        "false_reject_rate": mean(state == "reject" for state in states),
        "oos_reject_rate": mean(state == "reject" for state in oos_states) if oos_states else None,
        "clarification_rate": mean(state == "clarify" for state in states),
    }
