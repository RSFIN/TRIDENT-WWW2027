from __future__ import annotations

from mulaconf_baseline.metrics import jaccard, multilabel_point_metrics, roc_auc


def test_jaccard() -> None:
    assert jaccard(("a", "b"), ("b", "c")) == 1 / 3


def test_multilabel_point_metrics() -> None:
    result = multilabel_point_metrics(
        [("a", "b"), ("c",)],
        [("a", "b"), ("a",)],
        [["a", "b", "c"], ["a", "c", "b"]],
    )
    assert result["exact_match"] == 0.5
    assert result["hit_at_3"] == 1.0


def test_roc_auc_perfect_ranking() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
