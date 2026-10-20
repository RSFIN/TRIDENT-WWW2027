"""Split-conformal predictors used by the CICC retrieval baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the ceil((n+1)(1-alpha)) finite-sample conformal quantile."""
    if not scores:
        raise ValueError("calibration scores cannot be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    ordered = sorted(float(score) for score in scores)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    if rank > len(ordered):
        return math.inf
    rank = max(rank, 1)
    return ordered[rank - 1]


def _validate_probability_rows(rows: Sequence[Mapping[str, float]], labels: Sequence[str]) -> None:
    expected = set(labels)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"probability row {index} does not match the label space")
        if any(value < 0.0 for value in row.values()):
            raise ValueError(f"probability row {index} contains a negative value")


@dataclass
class MarginalConformal:
    """CICC marginal split conformal with nonconformity 1 - p(true label)."""

    labels: list[str]
    alpha: float
    threshold: float | None = None

    def fit(self, probability_rows: Sequence[Mapping[str, float]], true_labels: Sequence[str]) -> "MarginalConformal":
        if len(probability_rows) != len(true_labels):
            raise ValueError("probabilities and labels must have equal length")
        _validate_probability_rows(probability_rows, self.labels)
        scores = [1.0 - row[label] for row, label in zip(probability_rows, true_labels)]
        self.threshold = finite_sample_quantile(scores, self.alpha)
        return self

    def predict_set(self, probabilities: Mapping[str, float], threshold: float | None = None) -> list[str]:
        cutoff = self.threshold if threshold is None else threshold
        if cutoff is None:
            raise RuntimeError("conformal predictor must be fit before prediction")
        return sorted(
            (label for label in self.labels if 1.0 - probabilities[label] <= cutoff),
            key=lambda label: (-probabilities[label], label),
        )


@dataclass
class APSConformal:
    """Adaptive prediction sets using cumulative ranked probability mass."""

    labels: list[str]
    alpha: float
    threshold: float | None = None

    @staticmethod
    def _ordered(probabilities: Mapping[str, float]) -> list[tuple[str, float]]:
        return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))

    @classmethod
    def score(cls, probabilities: Mapping[str, float], true_label: str) -> float:
        cumulative = 0.0
        for label, probability in cls._ordered(probabilities):
            cumulative += probability
            if label == true_label:
                return cumulative
        raise ValueError(f"true label {true_label!r} is outside the probability row")

    def fit(self, probability_rows: Sequence[Mapping[str, float]], true_labels: Sequence[str]) -> "APSConformal":
        if len(probability_rows) != len(true_labels):
            raise ValueError("probabilities and labels must have equal length")
        _validate_probability_rows(probability_rows, self.labels)
        scores = [self.score(row, label) for row, label in zip(probability_rows, true_labels)]
        self.threshold = finite_sample_quantile(scores, self.alpha)
        return self

    def predict_set(self, probabilities: Mapping[str, float], threshold: float | None = None) -> list[str]:
        cutoff = self.threshold if threshold is None else threshold
        if cutoff is None:
            raise RuntimeError("conformal predictor must be fit before prediction")
        selected: list[str] = []
        cumulative = 0.0
        for label, probability in self._ordered(probabilities):
            selected.append(label)
            cumulative += probability
            if cumulative >= cutoff:
                break
        return selected


def weighted_quantile_with_test_mass(
    scores: Sequence[float],
    calibration_weights: Sequence[float],
    test_weight: float,
    alpha: float,
) -> float:
    """Weighted split-conformal quantile with the test point represented at +infinity."""
    if len(scores) != len(calibration_weights) or not scores:
        raise ValueError("scores and non-empty calibration_weights must have equal length")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in calibration_weights):
        raise ValueError("calibration weights must be finite and non-negative")
    if test_weight < 0.0 or not math.isfinite(test_weight):
        raise ValueError("test_weight must be finite and non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    total = sum(calibration_weights) + test_weight
    if total <= 0.0:
        raise ValueError("the total conformal weight must be positive")
    target = (1.0 - alpha) * total
    cumulative = 0.0
    for score, weight in sorted(zip(scores, calibration_weights), key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= target:
            return float(score)
    return math.inf


def selection_likelihood_ratio(
    selected_probabilities: Sequence[float], baseline_probabilities: Sequence[float]
) -> float:
    """Compute CLARA's product of selection probability / baseline probability."""
    if len(selected_probabilities) != len(baseline_probabilities):
        raise ValueError("selection and baseline probabilities must have equal length")
    log_ratio = 0.0
    for selected, baseline in zip(selected_probabilities, baseline_probabilities):
        if not 0.0 < selected <= 1.0 or not 0.0 < baseline <= 1.0:
            raise ValueError("all selection probabilities must lie in (0, 1]")
        log_ratio += math.log(selected) - math.log(baseline)
    return math.exp(log_ratio)


@dataclass
class TurnValidWeightedConformal:
    """CLARA-style selection-reweighted APS for post-selection rounds.

    The caller must supply likelihood-ratio weights from a correctly specified
    selection model. This class deliberately does not invent those weights.
    """

    labels: list[str]
    alpha: float
    calibration_scores: list[float]

    @classmethod
    def from_calibration(
        cls,
        labels: Sequence[str],
        alpha: float,
        probability_rows: Sequence[Mapping[str, float]],
        true_labels: Sequence[str],
    ) -> "TurnValidWeightedConformal":
        _validate_probability_rows(probability_rows, labels)
        scores = [APSConformal.score(row, label) for row, label in zip(probability_rows, true_labels)]
        return cls(list(labels), alpha, scores)

    def threshold(self, calibration_likelihood_ratios: Sequence[float], test_likelihood_ratio: float) -> float:
        return weighted_quantile_with_test_mass(
            self.calibration_scores,
            calibration_likelihood_ratios,
            test_likelihood_ratio,
            self.alpha,
        )

    def predict_set(
        self,
        probabilities: Mapping[str, float],
        calibration_likelihood_ratios: Sequence[float],
        test_likelihood_ratio: float,
    ) -> list[str]:
        threshold = self.threshold(calibration_likelihood_ratios, test_likelihood_ratio)
        if math.isinf(threshold):
            return sorted(self.labels, key=lambda label: (-probabilities[label], label))
        predictor = APSConformal(self.labels, self.alpha, threshold=threshold)
        return predictor.predict_set(probabilities)

