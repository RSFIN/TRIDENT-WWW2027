"""Training-free TF-IDF labelset prototype scorer."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .data import Example, LabelSet


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().replace("_", " "))


@dataclass(frozen=True)
class ScoreResult:
    probabilities: np.ndarray
    latency_ms: float
    token_count: int


class TfidfLabelsetRetriever:
    """One sparse TF-IDF centroid per observed labelset; no gradient fitting."""

    parameter_count = 0

    def __init__(self, *, temperature: float = 0.07, label_name_weight: float = 2.0) -> None:
        self.temperature = temperature
        self.label_name_weight = label_name_weight
        self.classes: list[LabelSet] = []
        self.idf: dict[str, float] = {}
        self.centroids: list[dict[str, float]] = []
        self.index_build_seconds = 0.0

    @staticmethod
    def _normalize(vector: Mapping[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {} if norm == 0 else {key: value / norm for key, value in vector.items()}

    def fit(self, examples: Sequence[Example]) -> "TfidfLabelsetRetriever":
        started = perf_counter()
        document_frequency: Counter[str] = Counter()
        tokenized: list[tuple[Example, list[str]]] = []
        for example in examples:
            if example.is_oos:
                continue
            tokens = tokenize(example.text)
            tokenized.append((example, tokens))
            document_frequency.update(set(tokens))
        if not tokenized:
            raise ValueError("no in-scope examples available for indexing")
        n = len(tokenized)
        self.idf = {
            token: math.log((1.0 + n) / (1.0 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        sums: dict[LabelSet, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Counter[LabelSet] = Counter()
        for example, tokens in tokenized:
            for token, count in Counter(tokens).items():
                sums[example.labels][token] += (1.0 + math.log(count)) * self.idf[token]
            counts[example.labels] += 1
        self.classes = sorted(sums)
        self.centroids = []
        for labelset in self.classes:
            vector = {token: value / counts[labelset] for token, value in sums[labelset].items()}
            for token, count in Counter(tokenize(" ".join(labelset))).items():
                vector[token] = vector.get(token, 0.0) + self.label_name_weight * (
                    1.0 + math.log(count)
                ) * self.idf.get(token, math.log(1.0 + n))
            self.centroids.append(self._normalize(vector))
        self.index_build_seconds = perf_counter() - started
        return self

    def predict(self, text: str) -> ScoreResult:
        started = perf_counter()
        tokens = tokenize(text)
        raw = {
            token: (1.0 + math.log(count)) * self.idf[token]
            for token, count in Counter(tokens).items()
            if token in self.idf
        }
        query = self._normalize(raw)
        similarities = np.asarray(
            [sum(value * centroid.get(token, 0.0) for token, value in query.items()) for centroid in self.centroids],
            dtype=float,
        )
        logits = (similarities - similarities.max()) / self.temperature
        exponentials = np.exp(logits)
        probabilities = exponentials / exponentials.sum()
        return ScoreResult(probabilities, (perf_counter() - started) * 1000.0, len(tokens))
