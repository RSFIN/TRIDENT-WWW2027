"""End-to-end CLINC150 experiment runner for training-free CICC."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .conformal import APSConformal, MarginalConformal
from .data import (
    Example,
    limit_per_label,
    load_clinc150,
    load_hard_negative_jsonl,
    load_unified_clinc150,
    stratified_partition,
)
from .decision import DecisionType, decide, oos_score
from .display import build_panel
from .metrics import (
    accuracy,
    average_precision,
    binary_f1,
    empirical_coverage,
    expected_calibration_error,
    fpr_at_recall,
    macro_f1,
    multiclass_brier,
    percentile,
    roc_auc,
    safe_divide,
    set_size_distribution,
    threshold_for_recall,
)
from .retrieval import RetrievalResult, TfidfIntentRetriever
from .statistics import summarize_runs
from .unified_v1 import compute_unified_v1


@dataclass(frozen=True)
class ScoredExample:
    example: Example
    retrieval: RetrievalResult


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _jsonl_dump(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_id(split: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"clinc150\0{split}\0{index}\0{text}".encode("utf-8")).hexdigest()
    return digest[:20]


def _build_conformal(method: str, labels: list[str], alpha: float):
    normalized = method.lower()
    if normalized == "aps":
        return APSConformal(labels, alpha)
    if normalized == "marginal":
        return MarginalConformal(labels, alpha)
    raise ValueError(f"unsupported conformal_method: {method!r}")


def _score_examples(retriever: TfidfIntentRetriever, examples: Sequence[Example]) -> list[ScoredExample]:
    return [ScoredExample(example, retriever.predict(example.text)) for example in examples]


def _fit_conformal(
    method: str,
    labels: list[str],
    alpha: float,
    calibration: Sequence[ScoredExample],
):
    predictor = _build_conformal(method, labels, alpha)
    predictor.fit(
        [item.retrieval.probabilities for item in calibration],
        [item.example.label for item in calibration],
    )
    return predictor


def _select_operating_point(
    *,
    method: str,
    labels: list[str],
    calibration: Sequence[ScoredExample],
    tuning_in_scope: Sequence[ScoredExample],
    tuning_oos: Sequence[ScoredExample],
    alpha_grid: Sequence[float],
    clarification_size_grid: Sequence[int],
    target_oos_recall: float,
) -> dict[str, object]:
    tuning = list(tuning_in_scope) + list(tuning_oos)
    y_oos = [int(item.example.is_oos) for item in tuning]
    candidates: list[dict[str, object]] = []
    best_key: tuple[float, float, int, float] | None = None
    best: dict[str, object] | None = None

    for alpha in alpha_grid:
        predictor = _fit_conformal(method, labels, float(alpha), calibration)
        sets = [predictor.predict_set(item.retrieval.probabilities) for item in tuning]
        scores = [1.0 - max(item.retrieval.probabilities.values()) for item in tuning]
        recall_threshold = threshold_for_recall(y_oos, scores, target_oos_recall)
        for max_size in clarification_size_grid:
            decisions = [
                decide(candidate_set, item.retrieval.probabilities, int(max_size))
                for candidate_set, item in zip(sets, tuning)
            ]
            rejected = [int(decision.action == DecisionType.REJECT) for decision in decisions]
            f1 = binary_f1(y_oos, rejected)
            average_size = mean(len(candidate_set) for candidate_set in sets)
            row = {
                "alpha": float(alpha),
                "max_clarification_size": int(max_size),
                "oos_f1": f1,
                "average_set_size": average_size,
                "oos_score_threshold_for_target_recall": recall_threshold,
            }
            candidates.append(row)
            key = (f1, -average_size, -int(max_size), -float(alpha))
            if best_key is None or key > best_key:
                best_key = key
                best = row
    assert best is not None
    return {"selected": best, "grid": candidates}


def _evaluate_repeat(
    *,
    seed: int,
    config: Mapping[str, Any],
    retriever: TfidfIntentRetriever,
    val: Sequence[Example],
    oos_val: Sequence[Example],
    test: Sequence[Example],
    oos_test: Sequence[Example],
    hard_negative_test: Sequence[Example],
    output_dir: Path,
) -> dict[str, object]:
    if config.get("protocol_version") == "unified_v1.0":
        calibration_examples, tuning_examples = list(val), list(val)
    else:
        calibration_examples, tuning_examples = stratified_partition(
            val, float(config["calibration_fraction"]), seed
        )
    calibration = _score_examples(retriever, calibration_examples)
    tuning_in_scope = _score_examples(retriever, tuning_examples)
    tuning_oos = _score_examples(retriever, oos_val)

    selection = _select_operating_point(
        method=str(config["conformal_method"]),
        labels=retriever.labels,
        calibration=calibration,
        tuning_in_scope=tuning_in_scope,
        tuning_oos=tuning_oos,
        alpha_grid=[float(value) for value in config["alpha_grid"]],
        clarification_size_grid=[int(value) for value in config["max_clarification_size_grid"]],
        target_oos_recall=float(config["target_oos_recall"]),
    )
    selected = selection["selected"]
    alpha = float(selected["alpha"])
    max_size = int(selected["max_clarification_size"])
    tuned_oos_threshold = float(selected["oos_score_threshold_for_target_recall"])
    predictor = _fit_conformal(
        str(config["conformal_method"]), retriever.labels, alpha, calibration
    )

    in_scope_scored = _score_examples(retriever, test)
    oos_scored = _score_examples(retriever, oos_test)
    scored = in_scope_scored + oos_scored
    candidate_sets = [predictor.predict_set(item.retrieval.probabilities) for item in scored]
    decisions = [
        decide(candidate_set, item.retrieval.probabilities, max_size)
        for candidate_set, item in zip(candidate_sets, scored)
    ]
    top1 = [max(item.retrieval.probabilities, key=item.retrieval.probabilities.get) for item in scored]
    confidences = [max(item.retrieval.probabilities.values()) for item in scored]
    oos_scores = [1.0 - confidence for confidence in confidences]
    y_oos = [int(item.example.is_oos) for item in scored]
    rejected = [int(decision.action == DecisionType.REJECT) for decision in decisions]
    tuned_rejected = [int(score >= tuned_oos_threshold) for score in oos_scores]

    n_in_scope = len(in_scope_scored)
    in_scope_sets = candidate_sets[:n_in_scope]
    in_scope_decisions = decisions[:n_in_scope]
    in_scope_top1 = top1[:n_in_scope]
    in_scope_confidences = confidences[:n_in_scope]
    in_scope_probabilities = [item.retrieval.probabilities for item in in_scope_scored]
    true_labels = [item.example.label for item in in_scope_scored]
    correctness = [truth == prediction for truth, prediction in zip(true_labels, in_scope_top1)]
    ece, reliability = expected_calibration_error(
        correctness, in_scope_confidences, int(config["ece_bins"])
    )
    coverage = empirical_coverage(in_scope_sets, true_labels)
    clarification_indices = [
        index
        for index, decision in enumerate(in_scope_decisions)
        if decision.action == DecisionType.CLARIFY
    ]
    clarification_coverage = (
        mean(true_labels[index] in in_scope_sets[index] for index in clarification_indices)
        if clarification_indices
        else None
    )
    final_sets = [
        tuple(candidate_set) if decision.action != DecisionType.REJECT else ()
        for candidate_set, decision in zip(in_scope_sets, in_scope_decisions)
    ]
    states = [
        "reject" if decision.action == DecisionType.REJECT
        else "clarify" if decision.action == DecisionType.CLARIFY
        else "release"
        for decision in in_scope_decisions
    ]
    oos_states = [
        "reject" if decision.action == DecisionType.REJECT
        else "clarify" if decision.action == DecisionType.CLARIFY
        else "release"
        for decision in decisions[n_in_scope:]
    ]
    unified = compute_unified_v1(
        gold_sets=[(label,) for label in true_labels],
        final_sets=final_sets,
        rankings=final_sets,
        confidences=in_scope_confidences,
        states=states,
        oos_states=oos_states,
    )
    latencies = [item.retrieval.latency_ms for item in scored]
    token_counts = [item.retrieval.token_count for item in scored]

    hard_negative_scored = _score_examples(retriever, hard_negative_test)
    hard_negative_sets = [
        predictor.predict_set(item.retrieval.probabilities) for item in hard_negative_scored
    ]
    hard_negative_decisions = [
        decide(candidate_set, item.retrieval.probabilities, max_size)
        for candidate_set, item in zip(hard_negative_sets, hard_negative_scored)
    ]
    hard_negative_scores = [
        1.0 - max(item.retrieval.probabilities.values()) for item in hard_negative_scored
    ]
    latencies.extend(item.retrieval.latency_ms for item in hard_negative_scored)
    token_counts.extend(item.retrieval.token_count for item in hard_negative_scored)
    if hard_negative_scored:
        hard_negative_y = [0] * n_in_scope + [1] * len(hard_negative_scored)
        hard_negative_ranking_scores = oos_scores[:n_in_scope] + hard_negative_scores
        hard_negative_metrics: dict[str, object] = {
            "evaluated": True,
            "count": len(hard_negative_scored),
            "auroc_vs_in_scope": roc_auc(hard_negative_y, hard_negative_ranking_scores),
            "auprc_vs_in_scope": average_precision(hard_negative_y, hard_negative_ranking_scores),
            "cicc_reject_recall": mean(
                decision.action == DecisionType.REJECT for decision in hard_negative_decisions
            ),
            "tuned_threshold_recall": mean(
                score >= tuned_oos_threshold for score in hard_negative_scores
            ),
            "average_prediction_set_size": mean(len(values) for values in hard_negative_sets),
        }
    else:
        hard_negative_metrics = {
            "evaluated": False,
            "count": 0,
            "reason": "hard_negative_path is null; do not interpret original CLINC150 OOS as hard-negative OOS",
        }

    metrics: dict[str, object] = {
        "seed": seed,
        "unified_v1": unified,
        "unified_v1_metadata": {
            "protocol": "v1.0",
            "alpha": alpha,
            "ece_confidence_source": "TF-IDF retriever top-1 probability",
            "reject_definition": "final prediction set P is empty",
        },
        "selected_operating_point": selected,
        "sample_counts": {
            "index_train": config["runtime_counts"]["train"],
            "conformal_calibration": len(calibration),
            "threshold_tuning_in_scope": len(tuning_in_scope),
            "threshold_tuning_oos": len(tuning_oos),
            "test_in_scope": len(in_scope_scored),
            "test_oos": len(oos_scored),
            "test_hard_negative_oos": len(hard_negative_scored),
        },
        "L1_closed": {
            "accuracy": accuracy(true_labels, in_scope_top1),
            "macro_f1": macro_f1(true_labels, in_scope_top1),
        },
        "L2_oos": {
            "auroc": roc_auc(y_oos, oos_scores),
            "auprc": average_precision(y_oos, oos_scores),
            "fpr95_test_operating_point": fpr_at_recall(
                y_oos, oos_scores, float(config["target_oos_recall"])
            ),
            "f1_oos_cicc_decision": binary_f1(y_oos, rejected),
            "f1_oos_tuned_recall_threshold": binary_f1(y_oos, tuned_rejected),
            "hard_negative_subset": hard_negative_metrics,
        },
        "L3_calibration": {
            "ece_15_bin": ece,
            "brier_multiclass": multiclass_brier(in_scope_probabilities, true_labels),
            "empirical_coverage": coverage,
            "nominal_coverage": 1.0 - alpha,
            "coverage_gap": coverage - (1.0 - alpha),
            "average_prediction_set_size": mean(len(values) for values in in_scope_sets),
            "prediction_set_size_distribution": set_size_distribution(in_scope_sets),
            "reliability_bins": reliability,
            "coverage_by_turn": {"0": coverage},
            "turn_valid_multi_round_evaluated": False,
            "turn_valid_reason": "CLINC150 provides no interactive selection traces; only first-turn split-conformal coverage is evaluated",
        },
        "L4_efficiency": {
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p99": percentile(latencies, 0.99),
            "whitespace_like_tokens_per_sample": mean(token_counts),
            "trainable_parameter_count": retriever.parameter_count,
            "index_build_seconds": retriever.index_build_seconds,
        },
        "L5_decision": {
            "cicc_reject_rate": mean(rejected),
            "cicc_false_reject_rate_in_scope": mean(rejected[:n_in_scope]),
            "reject_rate_at_tuned_95_oos_recall": mean(tuned_rejected),
            "false_reject_rate_at_tuned_95_oos_recall": mean(tuned_rejected[:n_in_scope]),
            "realized_oos_recall_at_tuned_threshold": mean(tuned_rejected[n_in_scope:]),
            "clarification_rate_in_scope": safe_divide(len(clarification_indices), n_in_scope),
            "clarification_set_coverage": clarification_coverage,
            "mean_clarification_rounds": safe_divide(len(clarification_indices), n_in_scope),
            "clarification_trigger_accuracy": None,
            "clarification_trigger_accuracy_reason": "CLINC150 has no ambiguity/clarification-need ground truth",
        },
    }

    repeat_dir = output_dir / f"seed_{seed}"
    prediction_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    panel_limit = int(config["clarification_panel_limit"])
    for index, (item, candidate_set, decision, predicted, confidence, uncertainty) in enumerate(
        zip(scored, candidate_sets, decisions, top1, confidences, oos_scores)
    ):
        split = "test_oos" if item.example.is_oos else "test"
        final_set = [] if decision.action == DecisionType.REJECT else list(candidate_set)
        prediction_rows.append(
            {
                "sample_id": _sample_id(split, index, item.example.text),
                "split": split,
                "true_label": item.example.label,
                "is_oos": item.example.is_oos,
                "top1_prediction": predicted,
                "top1_confidence": confidence,
                "candidate_set": list(candidate_set),
                "candidate_set_size": len(candidate_set),
                "final_set": final_set,
                "ranking": final_set,
                "action": decision.action.value,
                "oos_score": uncertainty,
                "top1_correct": (not item.example.is_oos and predicted == item.example.label),
                "unified_main_correct": (
                    not item.example.is_oos and bool(final_set) and final_set[0] == item.example.label
                ),
            }
        )
        if decision.action == DecisionType.CLARIFY and len(panel_rows) < panel_limit:
            panel = build_panel(
                item.example.text,
                candidate_set,
                item.retrieval.probabilities,
                retriever.representatives,
            ).to_dict()
            panel["sample_id"] = prediction_rows[-1]["sample_id"]
            panel["true_label"] = item.example.label
            panel_rows.append(panel)

    for index, (item, candidate_set, decision, uncertainty) in enumerate(
        zip(hard_negative_scored, hard_negative_sets, hard_negative_decisions, hard_negative_scores)
    ):
        probabilities = item.retrieval.probabilities
        predicted = max(probabilities, key=probabilities.get)
        prediction_rows.append(
            {
                "sample_id": _sample_id("test_hard_negative", index, item.example.text),
                "split": "test_hard_negative",
                "true_label": item.example.label,
                "is_oos": True,
                "top1_prediction": predicted,
                "top1_confidence": max(probabilities.values()),
                "candidate_set": list(candidate_set),
                "candidate_set_size": len(candidate_set),
                "action": decision.action.value,
                "oos_score": uncertainty,
                "top1_correct": False,
            }
        )

    _json_dump(repeat_dir / "metrics.json", metrics)
    _json_dump(repeat_dir / "operating_point_search.json", selection)
    _jsonl_dump(repeat_dir / "predictions.jsonl", prediction_rows)
    _jsonl_dump(repeat_dir / "clarification_panels.jsonl", panel_rows)
    return metrics


def run_experiment(config_path: str | Path, *, project_root: str | Path, smoke: bool = False) -> Path:
    root = Path(project_root).resolve()
    with Path(config_path).resolve().open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    data_path = root / config["data_path"]
    output_dir = root / config["output_dir"]
    if smoke:
        output_dir = output_dir / "smoke"
    splits = (
        load_unified_clinc150(data_path)
        if config.get("data_format") == "unified_clinc150"
        else load_clinc150(data_path)
    )

    train = splits.train
    val = splits.val
    oos_val = splits.oos_val
    test = splits.test
    oos_test = splits.oos_test
    hard_negative_path = config.get("hard_negative_path")
    hard_negative_test = (
        load_hard_negative_jsonl(root / str(hard_negative_path))
        if hard_negative_path
        else []
    )
    seeds = [int(value) for value in config["seeds"]]
    if smoke:
        train = limit_per_label(train, 5)
        val = limit_per_label(val, 4)
        oos_val = oos_val[:50]
        test = limit_per_label(test, 2)
        oos_test = oos_test[:50]
        hard_negative_test = hard_negative_test[:50]
        seeds = seeds[:1]

    retriever_config = config["retriever"]
    retriever = TfidfIntentRetriever(
        temperature=float(retriever_config["temperature"]),
        label_name_weight=float(retriever_config["label_name_weight"]),
        sublinear_tf=bool(retriever_config["sublinear_tf"]),
    ).fit(train)
    config["runtime_counts"] = {
        "train": len(train),
        "val": len(val),
        "oos_val": len(oos_val),
        "test": len(test),
        "oos_test": len(oos_test),
        "hard_negative_test": len(hard_negative_test),
    }

    runs = [
        _evaluate_repeat(
            seed=seed,
            config=config,
            retriever=retriever,
            val=val,
            oos_val=oos_val,
            test=test,
            oos_test=oos_test,
            hard_negative_test=hard_negative_test,
            output_dir=output_dir,
        )
        for seed in seeds
    ]
    summary = {
        "experiment": "training-free CICC on CLINC150",
        "smoke": smoke,
        "seeds": seeds,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "environment_manager": "not configured; reserved for user-managed uv setup",
        },
        "config": config,
        "repeat_summary": summarize_runs(runs),
    }
    _json_dump(output_dir / "summary.json", summary)
    return output_dir

