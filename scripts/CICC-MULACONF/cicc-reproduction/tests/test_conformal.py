from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cicc_baseline.conformal import (  # noqa: E402
    APSConformal,
    MarginalConformal,
    finite_sample_quantile,
    selection_likelihood_ratio,
    weighted_quantile_with_test_mass,
)


class ConformalTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_ceiling_correction(self) -> None:
        self.assertEqual(finite_sample_quantile([0.1, 0.2, 0.3, 0.4], 0.25), 0.4)

    def test_finite_sample_quantile_is_conservative_when_rank_exceeds_n(self) -> None:
        self.assertEqual(finite_sample_quantile([0.1, 0.2], 0.01), float("inf"))

    def test_marginal_prediction_set(self) -> None:
        predictor = MarginalConformal(["a", "b", "c"], alpha=0.2)
        predictor.fit(
            [
                {"a": 0.8, "b": 0.1, "c": 0.1},
                {"a": 0.1, "b": 0.8, "c": 0.1},
                {"a": 0.1, "b": 0.1, "c": 0.8},
                {"a": 0.7, "b": 0.2, "c": 0.1},
            ],
            ["a", "b", "c", "a"],
        )
        self.assertEqual(predictor.predict_set({"a": 0.75, "b": 0.2, "c": 0.05}), ["a"])

    def test_aps_returns_smallest_prefix_crossing_threshold(self) -> None:
        predictor = APSConformal(["a", "b", "c"], alpha=0.1, threshold=0.8)
        selected = predictor.predict_set({"a": 0.5, "b": 0.3, "c": 0.2})
        self.assertEqual(selected, ["a", "b"])

    def test_weighted_quantile_can_return_infinity_for_large_test_mass(self) -> None:
        threshold = weighted_quantile_with_test_mass(
            [0.1, 0.2], [1.0, 1.0], test_weight=100.0, alpha=0.1
        )
        self.assertEqual(threshold, float("inf"))

    def test_selection_likelihood_ratio_multiplies_turn_ratios(self) -> None:
        ratio = selection_likelihood_ratio([0.8, 0.6], [0.4, 0.3])
        self.assertAlmostEqual(ratio, 4.0)


if __name__ == "__main__":
    unittest.main()

