"""Repeated-run summaries and paired exact McNemar testing."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Mapping, Sequence


T_975_DF2 = 4.3026527297


def summarize_scalar(values: Sequence[float]) -> dict[str, float | int]:
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
    result: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[name] = float(value)
        elif isinstance(value, Mapping):
            result.update(flatten_numeric(value, name))
    return result


def summarize_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    flattened = [flatten_numeric(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in flattened)))
    return {key: summarize_scalar([row[key] for row in flattened if key in row]) for key in keys}


def mcnemar_exact(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> dict[str, float | int]:
    if len(first_correct) != len(second_correct) or not first_correct:
        raise ValueError("paired correctness vectors must be non-empty and equal length")
    first_only = sum(a and not b for a, b in zip(first_correct, second_correct))
    second_only = sum(b and not a for a, b in zip(first_correct, second_correct))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant": discordant,
        "p_value_exact_two_sided": p_value,
    }
