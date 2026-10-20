from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cicc_baseline.metrics import (  # noqa: E402
    accuracy,
    average_precision,
    binary_f1,
    macro_f1,
    roc_auc,
)
from cicc_baseline.statistics import mcnemar_exact  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        truth = ["a", "a", "b", "b"]
        predicted = ["a", "b", "b", "b"]
        self.assertAlmostEqual(accuracy(truth, predicted), 0.75)
        self.assertAlmostEqual(macro_f1(truth, predicted), (2 / 3 + 0.8) / 2)

    def test_binary_ranking_metrics(self) -> None:
        truth = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(roc_auc(truth, scores), 1.0)
        self.assertEqual(average_precision(truth, scores), 1.0)
        self.assertEqual(binary_f1(truth, [0, 0, 1, 1]), 1.0)

    def test_mcnemar_exact(self) -> None:
        result = mcnemar_exact(
            ["a", "b", "c", "d"],
            ["a", "x", "c", "x"],
            ["x", "b", "c", "d"],
        )
        self.assertEqual(result["a_correct_b_wrong"], 1)
        self.assertEqual(result["a_wrong_b_correct"], 2)
        self.assertEqual(result["discordant"], 3)


if __name__ == "__main__":
    unittest.main()

