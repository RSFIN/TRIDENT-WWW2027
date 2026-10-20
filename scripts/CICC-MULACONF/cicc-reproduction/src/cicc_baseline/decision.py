"""CICC's set-size driven action policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class DecisionType(str, Enum):
    ACCEPT = "accept"
    CLARIFY = "clarify"
    REJECT = "reject"


@dataclass(frozen=True)
class Decision:
    action: DecisionType
    predicted_label: str | None
    candidates: tuple[str, ...]
    reason: str


def decide(
    candidates: Sequence[str],
    probabilities: Mapping[str, float],
    max_clarification_size: int,
) -> Decision:
    if max_clarification_size < 2:
        raise ValueError("max_clarification_size must be at least 2")
    ordered = tuple(sorted(candidates, key=lambda label: (-probabilities[label], label)))
    if len(ordered) == 1:
        return Decision(DecisionType.ACCEPT, ordered[0], ordered, "singleton conformal set")
    if len(ordered) == 0:
        return Decision(DecisionType.REJECT, None, ordered, "empty conformal set")
    if len(ordered) > max_clarification_size:
        return Decision(
            DecisionType.REJECT,
            None,
            ordered,
            f"set size exceeds clarification limit {max_clarification_size}",
        )
    return Decision(DecisionType.CLARIFY, None, ordered, "non-singleton conformal set")


def oos_score(candidates: Sequence[str], probabilities: Mapping[str, float]) -> float:
    """Continuous CICC uncertainty: set size with max-probability tie breaking."""
    max_probability = max(probabilities.values()) if probabilities else 0.0
    return float(len(candidates)) + (1.0 - max_probability) / 2.0

