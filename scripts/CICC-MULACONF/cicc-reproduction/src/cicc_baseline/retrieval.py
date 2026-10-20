"""A dependency-free, training-free intent prototype retriever."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from .data import Example


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().replace("_", " "))


@dataclass(frozen=True)
class RetrievalResult:
    probabilities: dict[str, float]
    similarities: dict[str, float]
    latency_ms: float
    token_count: int


class TfidfIntentRetriever:
    """Builds one normalized TF-IDF centroid per intent without gradient training."""

    parameter_count = 0

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        label_name_weight: float = 2.0,
        sublinear_tf: bool = True,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.label_name_weight = label_name_weight
        self.sublinear_tf = sublinear_tf
        self.labels: list[str] = []
        self.idf: dict[str, float] = {}
        self.centroids: dict[str, dict[str, float]] = {}
        self.representatives: dict[str, str] = {}
        self.index_build_seconds = 0.0

    def _tf(self, count: int) -> float:
        return 1.0 + math.log(count) if self.sublinear_tf and count > 0 else float(count)

    @staticmethod
    def _normalize(vector: Mapping[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {token: value / norm for token, value in vector.items()}

    def fit(self, examples: Sequence[Example]) -> "TfidfIntentRetriever":
        started = perf_counter()
        if not examples:
            raise ValueError("at least one indexing example is required")
        document_frequencies: Counter[str] = Counter()
        tokenized: list[tuple[Example, list[str]]] = []
        for example in examples:
            if example.is_oos:
                continue
            tokens = tokenize(example.text)
            tokenized.append((example, tokens))
            document_frequencies.update(set(tokens))
            self.representatives.setdefault(example.label, example.text)
        n_documents = len(tokenized)
        self.idf = {
            token: math.log((1.0 + n_documents) / (1.0 + frequency)) + 1.0
            for token, frequency in document_frequencies.items()
        }

        sums: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Counter[str] = Counter()
        for example, tokens in tokenized:
            tf = Counter(tokens)
            for token, count in tf.items():
                sums[example.label][token] += self._tf(count) * self.idf.get(token, 0.0)
            counts[example.label] += 1

        for label in sorted(sums):
            averaged = {token: value / counts[label] for token, value in sums[label].items()}
            label_tokens = Counter(tokenize(label))
            for token, count in label_tokens.items():
                fallback_idf = math.log(1.0 + n_documents)
                averaged[token] = averaged.get(token, 0.0) + (
                    self.label_name_weight * self._tf(count) * self.idf.get(token, fallback_idf)
                )
            self.centroids[label] = self._normalize(averaged)
        self.labels = sorted(self.centroids)
        if not self.labels:
            raise ValueError("no in-scope labels were indexed")
        self.index_build_seconds = perf_counter() - started
        return self

    def _vectorize(self, text: str) -> tuple[dict[str, float], int]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        vector = {
            token: self._tf(count) * self.idf[token]
            for token, count in tf.items()
            if token in self.idf
        }
        return self._normalize(vector), len(tokens)

    def predict(self, text: str) -> RetrievalResult:
        if not self.labels:
            raise RuntimeError("retriever must be fit before prediction")
        started = perf_counter()
        query, token_count = self._vectorize(text)
        similarities: dict[str, float] = {}
        for label in self.labels:
            centroid = self.centroids[label]
            similarities[label] = sum(value * centroid.get(token, 0.0) for token, value in query.items())
        maximum = max(similarities.values())
        exponentials = {
            label: math.exp((similarity - maximum) / self.temperature)
            for label, similarity in similarities.items()
        }
        normalizer = sum(exponentials.values())
        probabilities = {label: value / normalizer for label, value in exponentials.items()}
        return RetrievalResult(
            probabilities=probabilities,
            similarities=similarities,
            latency_ms=(perf_counter() - started) * 1000.0,
            token_count=token_count,
        )

    def predict_many(self, texts: Iterable[str]) -> list[RetrievalResult]:
        return [self.predict(text) for text in texts]

