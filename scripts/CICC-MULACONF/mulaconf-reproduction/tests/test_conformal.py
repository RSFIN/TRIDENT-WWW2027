from __future__ import annotations

import numpy as np

from mulaconf_baseline.conformal import PowersetConformal, TreeConformal, conformal_p_value


CLASSES = [("a",), ("b",), ("a", "b")]


def test_conformal_p_value_uses_finite_sample_correction() -> None:
    scores = np.asarray([0.1, 0.2, 0.4])
    assert conformal_p_value(scores, 0.3) == 0.5


def test_powerset_region_contains_high_probability_class() -> None:
    model = PowersetConformal(CLASSES, alpha=0.1).fit(
        [np.asarray([0.8, 0.1, 0.1]), np.asarray([0.1, 0.8, 0.1])],
        [("a",), ("b",)],
    )
    assert ("a",) in model.predict_region(np.asarray([0.9, 0.05, 0.05]))


def test_tree_region_returns_known_labelsets() -> None:
    model = TreeConformal(CLASSES, alpha=0.1).fit(
        [
            np.asarray([0.8, 0.1, 0.1]),
            np.asarray([0.1, 0.8, 0.1]),
            np.asarray([0.1, 0.1, 0.8]),
        ],
        CLASSES,
    )
    region = model.predict_region(np.asarray([0.9, 0.05, 0.05]))
    assert region
    assert set(region).issubset(set(CLASSES))
