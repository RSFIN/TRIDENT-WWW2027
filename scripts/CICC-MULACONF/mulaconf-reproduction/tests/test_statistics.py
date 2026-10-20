from __future__ import annotations

from mulaconf_baseline.statistics import mcnemar_exact, summarize_scalar


def test_three_repeat_summary() -> None:
    result = summarize_scalar([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["ci95_low"] < 2.0 < result["ci95_high"]


def test_mcnemar_is_paired() -> None:
    result = mcnemar_exact([True, True, False], [False, True, True])
    assert result["discordant"] == 2
    assert result["p_value_exact_two_sided"] == 1.0
