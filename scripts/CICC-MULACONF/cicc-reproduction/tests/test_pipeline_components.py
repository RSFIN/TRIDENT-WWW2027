from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cicc_baseline.data import Example, stratified_partition  # noqa: E402
from cicc_baseline.decision import DecisionType, decide  # noqa: E402
from cicc_baseline.display import build_panel  # noqa: E402
from cicc_baseline.retrieval import TfidfIntentRetriever  # noqa: E402


class PipelineComponentTests(unittest.TestCase):
    def test_stratified_partition_keeps_each_label_on_both_sides(self) -> None:
        examples = [Example(f"a {index}", "a") for index in range(4)] + [
            Example(f"b {index}", "b") for index in range(4)
        ]
        calibration, tuning = stratified_partition(examples, 0.5, seed=42)
        self.assertEqual({item.label for item in calibration}, {"a", "b"})
        self.assertEqual({item.label for item in tuning}, {"a", "b"})
        self.assertEqual(len(calibration), 4)
        self.assertEqual(len(tuning), 4)

    def test_retriever_decision_and_panel(self) -> None:
        retriever = TfidfIntentRetriever(temperature=0.2).fit(
            [
                Example("book a flight ticket", "book_flight"),
                Example("reserve an airline ticket", "book_flight"),
                Example("cancel my bank transfer", "cancel_transfer"),
                Example("stop a money transfer", "cancel_transfer"),
            ]
        )
        result = retriever.predict("please book an airline flight")
        self.assertEqual(max(result.probabilities, key=result.probabilities.get), "book_flight")
        decision = decide(["book_flight", "cancel_transfer"], result.probabilities, 3)
        self.assertEqual(decision.action, DecisionType.CLARIFY)
        panel = build_panel(
            "please book an airline flight",
            decision.candidates,
            result.probabilities,
            retriever.representatives,
        )
        self.assertEqual(len(panel.cards), 2)


if __name__ == "__main__":
    unittest.main()

