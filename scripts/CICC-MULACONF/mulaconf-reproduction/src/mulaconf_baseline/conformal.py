"""Powerset and tree-structured split-conformal prediction regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.spatial.distance import pdist

from .data import LabelSet


def conformal_p_value(calibration_scores: np.ndarray, candidate_score: float) -> float:
    return float(
        (1 + np.count_nonzero(calibration_scores >= candidate_score))
        / (len(calibration_scores) + 1)
    )


class PowersetConformal:
    def __init__(self, classes: Sequence[LabelSet], alpha: float) -> None:
        self.classes = list(classes)
        self.alpha = alpha
        self.class_to_index = {value: index for index, value in enumerate(self.classes)}
        self.calibration_scores = np.asarray([], dtype=float)

    def fit(
        self, probabilities: Sequence[np.ndarray], labels: Sequence[LabelSet]
    ) -> "PowersetConformal":
        scores = []
        for probs, truth in zip(probabilities, labels):
            index = self.class_to_index.get(truth)
            scores.append(1.0 if index is None else 1.0 - float(probs[index]))
        self.calibration_scores = np.asarray(scores, dtype=float)
        return self

    def predict_region(self, probabilities: np.ndarray) -> list[LabelSet]:
        return [
            labelset
            for index, labelset in enumerate(self.classes)
            if conformal_p_value(
                self.calibration_scores, 1.0 - float(probabilities[index])
            )
            > self.alpha
        ]


@dataclass
class _TreeInfo:
    leaves: tuple[int, ...]
    calibration_scores: np.ndarray


class TreeConformal:
    """Average-linkage labelset tree with fixed-alpha hierarchical tests.

    A leaf survives only when every node on its path passes a split-conformal
    test at alpha divided by the maximum tree depth (hierarchical Bonferroni).
    """

    def __init__(self, classes: Sequence[LabelSet], alpha: float) -> None:
        self.classes = list(classes)
        self.alpha = alpha
        self.class_to_index = {value: index for index, value in enumerate(self.classes)}
        self.nodes: dict[int, _TreeInfo] = {}
        self.paths: dict[int, tuple[int, ...]] = {}
        self.max_depth = 1
        self._build_tree()

    def _build_tree(self) -> None:
        if len(self.classes) == 1:
            self.nodes[0] = _TreeInfo((0,), np.asarray([], dtype=float))
            self.paths[0] = (0,)
            return
        atomic = sorted(set().union(*map(set, self.classes)))
        matrix = np.asarray(
            [[int(label in labelset) for label in atomic] for labelset in self.classes],
            dtype=float,
        )
        root = to_tree(linkage(pdist(matrix, metric="hamming"), method="average"))

        def visit(node: ClusterNode, ancestors: tuple[int, ...]) -> tuple[int, ...]:
            path = ancestors + (node.id,)
            if node.is_leaf():
                leaves = (node.id,)
                self.paths[node.id] = path
                self.max_depth = max(self.max_depth, len(path))
            else:
                leaves = visit(node.left, path) + visit(node.right, path)
            self.nodes[node.id] = _TreeInfo(leaves, np.asarray([], dtype=float))
            return leaves

        visit(root, ())

    def fit(
        self, probabilities: Sequence[np.ndarray], labels: Sequence[LabelSet]
    ) -> "TreeConformal":
        collected: dict[int, list[float]] = {node_id: [] for node_id in self.nodes}
        for probs, truth in zip(probabilities, labels):
            leaf = self.class_to_index.get(truth)
            if leaf is None:
                continue
            for node_id in self.paths[leaf]:
                leaves = self.nodes[node_id].leaves
                collected[node_id].append(1.0 - float(probs[list(leaves)].sum()))
        for node_id, info in list(self.nodes.items()):
            self.nodes[node_id] = _TreeInfo(
                info.leaves, np.asarray(collected[node_id], dtype=float)
            )
        return self

    def predict_region(self, probabilities: np.ndarray) -> list[LabelSet]:
        local_alpha = self.alpha / self.max_depth
        node_passes: dict[int, bool] = {}
        for node_id, info in self.nodes.items():
            if len(info.calibration_scores) == 0:
                node_passes[node_id] = True
                continue
            score = 1.0 - float(probabilities[list(info.leaves)].sum())
            node_passes[node_id] = (
                conformal_p_value(info.calibration_scores, score) > local_alpha
            )
        accepted: list[LabelSet] = []
        for leaf, labelset in enumerate(self.classes):
            if all(node_passes[node_id] for node_id in self.paths[leaf]):
                accepted.append(labelset)
        return accepted
