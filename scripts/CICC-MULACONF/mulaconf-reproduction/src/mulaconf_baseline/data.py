"""Dataset loaders with explicit multi-label and OOS semantics."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LabelSet = tuple[str, ...]


@dataclass(frozen=True)
class Example:
    text: str
    labels: LabelSet
    is_oos: bool = False


@dataclass(frozen=True)
class DatasetSplits:
    name: str
    train: list[Example]
    val: list[Example]
    test: list[Example]
    oos_val: list[Example]
    oos_test: list[Example]


def canonical_labels(labels: Iterable[str]) -> LabelSet:
    return tuple(sorted(set(label.strip() for label in labels if label.strip())))


def _load_mix_file(path: Path) -> list[Example]:
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    examples: list[Example] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"invalid AGIF block in {path}")
        labels = canonical_labels(lines[-1].split("#"))
        tokens = [line.split(maxsplit=1)[0] for line in lines[:-1]]
        if not labels or not tokens:
            raise ValueError(f"empty text or labels in {path}")
        examples.append(Example(" ".join(tokens), labels))
    return examples


def load_mix_dataset(root: str | Path, name: str) -> DatasetSplits:
    directory = Path(root)
    return DatasetSplits(
        name=name,
        train=_load_mix_file(directory / "train.txt"),
        val=_load_mix_file(directory / "dev.txt"),
        test=_load_mix_file(directory / "test.txt"),
        oos_val=[],
        oos_test=[],
    )


def _clinc_rows(rows: Sequence[Sequence[str]], *, is_oos: bool) -> list[Example]:
    return [
        Example(str(text), ("__oos__",) if is_oos else canonical_labels([str(label)]), is_oos)
        for text, label in rows
    ]


def load_clinc150(path: str | Path) -> DatasetSplits:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if not required.issubset(payload):
        raise ValueError(f"CLINC150 is missing keys: {sorted(required - set(payload))}")
    return DatasetSplits(
        name="clinc150",
        train=_clinc_rows(payload["train"], is_oos=False),
        val=_clinc_rows(payload["val"], is_oos=False),
        test=_clinc_rows(payload["test"], is_oos=False),
        oos_val=_clinc_rows(payload["oos_val"], is_oos=True),
        oos_test=_clinc_rows(payload["oos_test"], is_oos=True),
    )


def _load_unified_split(path: Path) -> tuple[list[Example], list[Example]]:
    in_scope: list[Example] = []
    oos: list[Example] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        is_oos = bool(row["oos"])
        label = "__oos__" if is_oos else str(row["main_intent"])
        if not is_oos and not label:
            raise ValueError(f"missing main_intent at {path}:{line_number}")
        example = Example(str(row["utterance"]), (label,), is_oos)
        (oos if is_oos else in_scope).append(example)
    return in_scope, oos


def load_unified_clinc150(root: str | Path) -> DatasetSplits:
    directory = Path(root)
    train, _ = _load_unified_split(directory / "train.jsonl")
    calibration, oos_calibration = _load_unified_split(directory / "calib.jsonl")
    test, oos_test = _load_unified_split(directory / "test.jsonl")
    return DatasetSplits("clinc150", train, calibration, test, oos_calibration, oos_test)


def stratified_partition(
    examples: Sequence[Example], fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    groups: dict[LabelSet, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.labels].append(example)
    rng = random.Random(seed)
    first: list[Example] = []
    second: list[Example] = []
    for labelset in sorted(groups):
        rows = list(groups[labelset])
        rng.shuffle(rows)
        cut = max(1, min(len(rows) - 1, round(len(rows) * fraction))) if len(rows) > 1 else 1
        first.extend(rows[:cut])
        second.extend(rows[cut:])
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


def smoke_subset(splits: DatasetSplits) -> DatasetSplits:
    return DatasetSplits(
        name=splits.name,
        train=splits.train[:1500],
        val=splits.val[:400],
        test=splits.test[:300],
        oos_val=splits.oos_val[:50],
        oos_test=splits.oos_test[:100],
    )
