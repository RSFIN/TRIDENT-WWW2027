"""Repeated-run summaries and paired McNemar significance testing."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Mapping, Sequence


T_975_DF2 = 4.3026527297


def summarize_scalar_repeats(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    average = mean(finite)
    if len(finite) == 1:
        return {"n": 1, "mean": average, "std": 0.0, "ci95_low": average, "ci95_high": average}
    deviation = stdev(finite)
    multiplier = T_975_DF2 if len(finite) == 3 else 1.96
    half_width = multiplier * deviation / math.sqrt(len(finite))
    return {
        "n": len(finite),
        "mean": average,
        "std": deviation,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
    }


def flatten_numeric(payload: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flattened[name] = float(value)
        elif isinstance(value, Mapping):
            flattened.update(flatten_numeric(value, name))
    return flattened


def summarize_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    keys = sorted(set().union(*(flatten_numeric(run).keys() for run in runs)))
    flattened = [flatten_numeric(run) for run in runs]
    return {
        key: summarize_scalar_repeats([run[key] for run in flattened if key in run])
        for key in keys
    }


def _binomial_two_sided_equal_probability(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def mcnemar_exact(
    y_true: Sequence[str], predictions_a: Sequence[str], predictions_b: Sequence[str]
) -> dict[str, float | int]:
    if not (len(y_true) == len(predictions_a) == len(predictions_b)) or not y_true:
        raise ValueError("all non-empty McNemar inputs must have equal length")
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    for truth, first, second in zip(y_true, predictions_a, predictions_b):
        first_correct = first == truth
        second_correct = second == truth
        if first_correct and not second_correct:
            a_correct_b_wrong += 1
        elif second_correct and not first_correct:
            a_wrong_b_correct += 1
    discordant = a_correct_b_wrong + a_wrong_b_correct
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "p_value_exact_two_sided": _binomial_two_sided_equal_probability(
            min(a_correct_b_wrong, a_wrong_b_correct), discordant
        ),
    }

