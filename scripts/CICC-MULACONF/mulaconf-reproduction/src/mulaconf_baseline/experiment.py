"""End-to-end experiment orchestration for powerset/tree conformal baselines."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from .conformal import PowersetConformal, TreeConformal
from .data import DatasetSplits, Example, LabelSet, load_clinc150, load_mix_dataset, load_unified_clinc150, smoke_subset, stratified_partition
from .metrics import (
    average_precision,
    binary_f1,
    expected_calibration_error,
    fpr_at_recall,
    jaccard,
    multiclass_brier,
    multilabel_point_metrics,
    percentile,
    roc_auc,
    size_distribution,
    threshold_for_recall,
)
from .retrieval import ScoreResult, TfidfLabelsetRetriever
from .statistics import summarize_runs
from .unified_v1 import compute_unified_v1


@dataclass(frozen=True)
class ScoredExample:
    example: Example
    score: ScoreResult


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _jsonl_dump(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _sample_id(dataset: str, split: str, index: int, text: str) -> str:
    return hashlib.sha256(f"{dataset}\0{split}\0{index}\0{text}".encode()).hexdigest()[:20]


def _score(retriever: TfidfLabelsetRetriever, examples: Sequence[Example]) -> list[ScoredExample]:
    return [ScoredExample(example, retriever.predict(example.text)) for example in examples]


def _rank_atomic(probabilities: np.ndarray, classes: Sequence[LabelSet]) -> list[str]:
    scores: dict[str, float] = {}
    for probability, labelset in zip(probabilities, classes):
        for label in labelset:
            scores[label] = scores.get(label, 0.0) + float(probability)
    return sorted(scores, key=lambda label: (-scores[label], label))


def _decision(region: Sequence[LabelSet], max_clarification_size: int) -> str:
    if len(region) == 1:
        return "accept"
    if 1 < len(region) <= max_clarification_size:
        return "clarify"
    return "reject"


def _make_predictor(
    method: str,
    classes: Sequence[LabelSet],
    alpha: float,
    calibration: Sequence[ScoredExample],
):
    if method == "powerset":
        predictor = PowersetConformal(classes, alpha)
    elif method == "tree_fixed_alpha":
        predictor = TreeConformal(classes, alpha)
    else:
        raise ValueError(f"unsupported method: {method}")
    return predictor.fit(
        [item.score.probabilities for item in calibration],
        [item.example.labels for item in calibration],
    )


def _evaluate(
    *,
    dataset: str,
    method: str,
    seed: int,
    retriever: TfidfLabelsetRetriever,
    calibration: Sequence[ScoredExample],
    tuning: Sequence[ScoredExample],
    tuning_oos: Sequence[ScoredExample],
    test: Sequence[ScoredExample],
    test_oos: Sequence[ScoredExample],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, object]:
    alpha = float(config["alpha"])
    predictor = _make_predictor(method, retriever.classes, alpha, calibration)
    all_test = list(test) + list(test_oos)
    regions = [predictor.predict_region(item.score.probabilities) for item in all_test]
    point_predictions = [retriever.classes[int(np.argmax(item.score.probabilities))] for item in test]
    ranked_labels = [_rank_atomic(item.score.probabilities, retriever.classes) for item in test]
    truths = [item.example.labels for item in test]
    point = multilabel_point_metrics(truths, point_predictions, ranked_labels)
    truth_indices = [retriever.classes.index(value) if value in retriever.classes else None for value in truths]
    confidences = [float(item.score.probabilities.max()) for item in test]
    correctness = [truth == prediction for truth, prediction in zip(truths, point_predictions)]
    ece, reliability = expected_calibration_error(correctness, confidences, int(config["ece_bins"]))
    in_scope_regions = regions[: len(test)]
    seen_mask = [truth in retriever.classes for truth in truths]
    exact_coverage = mean(truth in region for truth, region in zip(truths, in_scope_regions))
    seen_exact_coverage = mean(
        truth in region
        for truth, region, seen in zip(truths, in_scope_regions, seen_mask)
        if seen
    )
    jaccard_coverage = mean(
        any(jaccard(truth, candidate) >= 0.5 for candidate in region)
        for truth, region in zip(truths, in_scope_regions)
    )
    actions = [_decision(region, int(config["max_clarification_size"])) for region in regions]
    clarification_indices = [index for index, action in enumerate(actions[: len(test)]) if action == "clarify"]
    clarification_coverage = (
        mean(
            any(jaccard(truths[index], candidate) >= 0.5 for candidate in in_scope_regions[index])
            for index in clarification_indices
        )
        if clarification_indices
        else None
    )
    oos_scores = [1.0 - float(item.score.probabilities.max()) for item in all_test]

    atomic_rankings = [_rank_atomic(item.score.probabilities, retriever.classes) for item in all_test]
    raw_atomic_sets = [
        tuple(label for label in ranking if any(label in labelset for labelset in region))
        for ranking, region in zip(atomic_rankings, regions)
    ]
    final_sets = [() if action == "reject" else values for action, values in zip(actions, raw_atomic_sets)]
    states = ["release" if action == "accept" else action for action in actions]
    unified = compute_unified_v1(
        gold_sets=truths,
        final_sets=final_sets[: len(test)],
        rankings=final_sets[: len(test)],
        confidences=confidences,
        states=states[: len(test)],
        oos_states=states[len(test) :],
    )

    if test_oos:
        tuning_all = list(tuning) + list(tuning_oos)
        tuning_y = [0] * len(tuning) + [1] * len(tuning_oos)
        tuning_scores = [1.0 - float(item.score.probabilities.max()) for item in tuning_all]
        threshold = threshold_for_recall(tuning_y, tuning_scores, float(config["target_oos_recall"]))
        y_oos = [0] * len(test) + [1] * len(test_oos)
        threshold_rejected = [int(score >= threshold) for score in oos_scores]
        decision_rejected = [int(action == "reject") for action in actions]
        l2: dict[str, object] = {
            "evaluated": True,
            "auroc": roc_auc(y_oos, oos_scores),
            "auprc": average_precision(y_oos, oos_scores),
            "fpr95": fpr_at_recall(y_oos, oos_scores, float(config["target_oos_recall"])),
            "f1_oos_decision": binary_f1(y_oos, decision_rejected),
            "f1_oos_tuned_threshold": binary_f1(y_oos, threshold_rejected),
            "hard_negative_subset": {"evaluated": False, "count": 0, "reason": "no hard-negative path configured"},
        }
        threshold_decision = {
            "reject_rate_at_95_oos_recall": mean(threshold_rejected),
            "false_reject_rate_at_95_oos_recall": mean(threshold_rejected[: len(test)]),
            "realized_oos_recall": mean(threshold_rejected[len(test) :]),
            "oos_threshold": threshold,
        }
    else:
        l2 = {"evaluated": False, "reason": "dataset has no OOS split"}
        threshold_decision = {}

    latencies = [item.score.latency_ms for item in all_test]
    tokens = [item.score.token_count for item in all_test]
    metrics: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "unified_v1": unified,
        "unified_v1_metadata": {
            "protocol": "v1.0",
            "alpha": alpha,
            "ece_confidence_source": "labelset retriever top-1 probability",
            "reject_definition": "final prediction set P is empty",
        },
        "sample_counts": {
            "index_train": int(config["runtime_counts"]["train"]),
            "conformal_calibration": len(calibration),
            "threshold_tuning_in_scope": len(tuning),
            "threshold_tuning_oos": len(tuning_oos),
            "test_in_scope": len(test),
            "test_oos": len(test_oos),
            "observed_labelsets": len(retriever.classes),
        },
        "L1_multilabel": point,
        "L2_oos": l2,
        "L3_calibration": {
            "ece_15_bin": ece,
            "brier_multiclass_labelset": multiclass_brier(
                [item.score.probabilities for item in test], truth_indices
            ),
            "exact_labelset_coverage": exact_coverage,
            "seen_labelset_rate": mean(seen_mask),
            "exact_coverage_seen_labelsets": seen_exact_coverage,
            "jaccard_at_0_5_coverage": jaccard_coverage,
            "nominal_coverage": 1.0 - alpha,
            "exact_coverage_gap": exact_coverage - (1.0 - alpha),
            "average_prediction_region_size": mean(map(len, in_scope_regions)),
            "prediction_region_size_distribution": size_distribution(in_scope_regions),
            "reliability_bins": reliability,
        },
        "L4_efficiency": {
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p99": percentile(latencies, 0.99),
            "tokens_per_sample_approx": mean(tokens),
            "trainable_parameter_count": retriever.parameter_count,
            "index_build_seconds": retriever.index_build_seconds,
        },
        "L5_decision": {
            "reject_rate": mean(action == "reject" for action in actions),
            "false_reject_rate_in_scope": mean(action == "reject" for action in actions[: len(test)]),
            "clarification_rate_in_scope": len(clarification_indices) / len(test),
            "clarification_set_jaccard_coverage": clarification_coverage,
            "mean_clarification_rounds": len(clarification_indices) / len(test),
            "set_level_jaccard_error_rate": mean(
                jaccard(truth, prediction) < 0.5 for truth, prediction in zip(truths, point_predictions)
            ),
            "clarification_trigger_accuracy": None,
            "clarification_trigger_accuracy_reason": "datasets provide no ambiguity/clarification-need ground truth",
            **threshold_decision,
        },
    }

    repeat_dir = output_dir / dataset / method / f"seed_{seed}"
    rows = []
    for index, (item, region, action, final_set) in enumerate(zip(all_test, regions, actions, final_sets)):
        prediction = retriever.classes[int(np.argmax(item.score.probabilities))]
        rows.append(
            {
                "sample_id": _sample_id(dataset, "test_oos" if item.example.is_oos else "test", index, item.example.text),
                "query": item.example.text,
                "true_labels": list(item.example.labels),
                "is_oos": item.example.is_oos,
                "predicted_labels": list(prediction),
                "prediction_region": [list(value) for value in region],
                "prediction_region_size": len(region),
                "final_set": list(final_set),
                "ranking": list(final_set),
                "action": action,
                "confidence": float(item.score.probabilities.max()),
                "oos_score": 1.0 - float(item.score.probabilities.max()),
                "exact_match": not item.example.is_oos and prediction == item.example.labels,
                "unified_main_correct": (
                    not item.example.is_oos
                    and bool(final_set)
                    and final_set[0] == item.example.labels[0]
                ),
            }
        )
    _json_dump(repeat_dir / "metrics.json", metrics)
    _jsonl_dump(repeat_dir / "predictions.jsonl", rows)
    return metrics


def _load_dataset(root: Path, spec: Mapping[str, Any]) -> DatasetSplits:
    path = root / str(spec["path"])
    if spec["format"] == "agif_mix":
        return load_mix_dataset(path, str(spec["name"]))
    if spec["format"] == "clinc150":
        return load_clinc150(path)
    if spec["format"] == "unified_clinc150":
        return load_unified_clinc150(path)
    raise ValueError(f"unsupported dataset format: {spec['format']}")


def _select_dataset_specs(
    specs: Sequence[Mapping[str, Any]], dataset_filter: str | None
) -> list[Mapping[str, Any]]:
    if dataset_filter is None:
        return list(specs)
    selected = [spec for spec in specs if spec["name"] == dataset_filter]
    if not selected:
        available = ", ".join(sorted(str(spec["name"]) for spec in specs))
        raise ValueError(f"unknown dataset {dataset_filter!r}; available datasets: {available}")
    return selected


def run_experiment(
    config_path: str | Path,
    *,
    project_root: str | Path,
    smoke: bool = False,
    dataset_filter: str | None = None,
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    output_dir = root / config["output_dir"]
    if smoke:
        output_dir = output_dir / "smoke"
    seeds = [int(seed) for seed in config["seeds"]]
    if smoke:
        seeds = seeds[:1]
    all_runs: dict[str, dict[str, list[dict[str, object]]]] = {}
    dataset_metadata: dict[str, object] = {}

    selected_specs = _select_dataset_specs(config["datasets"], dataset_filter)
    for spec in selected_specs:
        splits = _load_dataset(root, spec)
        if smoke:
            splits = smoke_subset(splits)
        retriever = TfidfLabelsetRetriever(**config["retriever"]).fit(splits.train)
        runtime_counts = {
            "train": len(splits.train), "val": len(splits.val), "test": len(splits.test),
            "oos_val": len(splits.oos_val), "oos_test": len(splits.oos_test),
        }
        local_config = dict(config)
        local_config["runtime_counts"] = runtime_counts
        dataset_metadata[splits.name] = {
            **runtime_counts,
            "observed_labelsets": len(retriever.classes),
            "atomic_labels": len(set().union(*(set(value) for value in retriever.classes))),
            "clinc_label_semantics": "singleton labelsets" if splits.name == "clinc150" else "native multi-intent labelsets",
        }
        all_runs[splits.name] = {method: [] for method in config["methods"]}
        for seed in seeds:
            if spec.get("format") == "unified_clinc150":
                calibration_rows, tuning_rows = list(splits.val), list(splits.val)
            else:
                calibration_rows, tuning_rows = stratified_partition(
                    splits.val, float(config["calibration_fraction"]), seed
                )
            calibration = _score(retriever, calibration_rows)
            tuning = _score(retriever, tuning_rows)
            tuning_oos = _score(retriever, splits.oos_val)
            test = _score(retriever, splits.test)
            test_oos = _score(retriever, splits.oos_test)
            for method in config["methods"]:
                metrics = _evaluate(
                    dataset=splits.name, method=method, seed=seed, retriever=retriever,
                    calibration=calibration, tuning=tuning, tuning_oos=tuning_oos,
                    test=test, test_oos=test_oos, config=local_config, output_dir=output_dir,
                )
                all_runs[splits.name][method].append(metrics)

    summary = {
        "experiment": "powerset/tree multi-label conformal baselines",
        "smoke": smoke,
        "seeds": seeds,
        "environment": {"python": sys.version, "platform": platform.platform(), "manager": "uv"},
        "datasets": dataset_metadata,
        "config": config,
        "results": {
            dataset: {method: summarize_runs(runs) for method, runs in methods.items()}
            for dataset, methods in all_runs.items()
        },
    }
    summary_name = f"summary_{dataset_filter}.json" if dataset_filter else "summary.json"
    _json_dump(output_dir / summary_name, summary)
    return output_dir
