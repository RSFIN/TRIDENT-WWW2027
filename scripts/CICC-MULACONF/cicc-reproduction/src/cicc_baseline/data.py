"""Dataset loading and leakage-safe calibration/tuning splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    is_oos: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Clinc150Splits:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    oos_train: list[Example]
    oos_val: list[Example]
    oos_test: list[Example]


def _rows_to_examples(rows: Sequence[Sequence[str]], *, is_oos: bool) -> list[Example]:
    examples: list[Example] = []
    for index, row in enumerate(rows):
        if len(row) != 2:
            raise ValueError(f"CLINC150 row {index} must contain [text, label], got {row!r}")
        text, label = row
        examples.append(Example(str(text), str(label), is_oos=is_oos))
    return examples


def load_clinc150(path: str | Path) -> Clinc150Splits:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"CLINC150 file is missing splits: {sorted(missing)}")
    return Clinc150Splits(
        train=_rows_to_examples(payload["train"], is_oos=False),
        val=_rows_to_examples(payload["val"], is_oos=False),
        test=_rows_to_examples(payload["test"], is_oos=False),
        oos_train=_rows_to_examples(payload["oos_train"], is_oos=True),
        oos_val=_rows_to_examples(payload["oos_val"], is_oos=True),
        oos_test=_rows_to_examples(payload["oos_test"], is_oos=True),
    )


def _load_unified_split(path: Path) -> tuple[list[Example], list[Example]]:
    in_scope: list[Example] = []
    oos: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            is_oos = bool(row["oos"])
            label = "__oos__" if is_oos else str(row["main_intent"])
            if not is_oos and not label:
                raise ValueError(f"missing main_intent at {path}:{line_number}")
            example = Example(str(row["utterance"]), label, is_oos=is_oos)
            (oos if is_oos else in_scope).append(example)
    return in_scope, oos


def load_unified_clinc150(root: str | Path) -> Clinc150Splits:
    """Load the paper's fixed CLINC150 train/calib/test JSONL partition."""
    directory = Path(root)
    train, oos_train = _load_unified_split(directory / "train.jsonl")
    calibration, oos_calibration = _load_unified_split(directory / "calib.jsonl")
    test, oos_test = _load_unified_split(directory / "test.jsonl")
    return Clinc150Splits(train, calibration, test, oos_train, oos_calibration, oos_test)


def load_hard_negative_jsonl(path: str | Path) -> list[Example]:
    """Load public hard-negative OOS rows with at least a `text` field."""
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "text" not in row:
                raise ValueError(f"hard-negative row {line_number} must be an object with a text field")
            examples.append(
                Example(
                    text=str(row["text"]),
                    label=str(row.get("label", "oos_hard_negative")),
                    is_oos=True,
                    metadata={"hard_negative": True, "source": row.get("source")},
                )
            )
    if not examples:
        raise ValueError("hard-negative JSONL contains no examples")
    return examples


def stratified_partition(
    examples: Sequence[Example], calibration_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        if example.is_oos:
            raise ValueError("stratified_partition expects in-scope examples only")
        by_label.setdefault(example.label, []).append(example)

    rng = random.Random(seed)
    calibration: list[Example] = []
    tuning: list[Example] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        cut = round(len(group) * calibration_fraction)
        cut = min(max(cut, 1), len(group) - 1)
        calibration.extend(group[:cut])
        tuning.extend(group[cut:])
    rng.shuffle(calibration)
    rng.shuffle(tuning)
    return calibration, tuning


def limit_per_label(examples: Iterable[Example], limit: int) -> list[Example]:
    counts: dict[tuple[str, bool], int] = {}
    selected: list[Example] = []
    for example in examples:
        key = (example.label, example.is_oos)
        if counts.get(key, 0) >= limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(example)
    return selected

