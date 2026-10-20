"""CLARA-style show-don't-ask panels for intent alternatives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class IntentCard:
    option: int
    intent: str
    score: float
    example: str | None


@dataclass(frozen=True)
class ClarificationPanel:
    query: str
    instruction: str
    cards: tuple[IntentCard, ...]
    coverage_source: str = "split-conformal candidate set"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cards"] = [asdict(card) for card in self.cards]
        return payload


def build_panel(
    query: str,
    candidates: Sequence[str],
    probabilities: Mapping[str, float],
    representatives: Mapping[str, str] | None = None,
) -> ClarificationPanel:
    ordered = sorted(candidates, key=lambda label: (-probabilities[label], label))
    cards = tuple(
        IntentCard(
            option=index + 1,
            intent=label,
            score=probabilities[label],
            example=None if representatives is None else representatives.get(label),
        )
        for index, label in enumerate(ordered)
    )
    return ClarificationPanel(
        query=query,
        instruction="请选择最符合真实需求的一个候选意图；若均不符合，请选择拒识。",
        cards=cards,
    )

