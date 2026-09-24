"""L1-L5 指标。"""
from typing import Dict, List
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def exact_match_set(true: List[List[str]], pred: List[List[str]]) -> float:
    """集合完全匹配率。"""
    n = sum(1 for t, p in zip(true, pred) if set(t) == set(p))
    return n / max(len(true), 1)


def set_jaccard(true: List[List[str]], pred: List[List[str]]) -> float:
    scores = []
    for t, p in zip(true, pred):
        st, sp = set(t), set(p)
        u = st | sp
        scores.append(len(st & sp) / len(u) if u else 1.0)
    return float(np.mean(scores))


def hit_at_k(true: List[List[str]], pred_ranked: List[List[str]], k: int) -> float:
    """true 中所有意图是否都在 pred_ranked 前 k 个里。"""
    hits = 0
    for t, p in zip(true, pred_ranked):
        topk = set(p[:k])
        if all(ti in topk for ti in t):
            hits += 1
    return hits / max(len(true), 1)


def oos_metrics(oos_scores: List[float], oos_labels: List[int]) -> Dict[str, float]:
    """oos_labels: 1=OOS, 0=in-domain。oos_scores 越大越像 OOS。"""
    return {
        "auroc": roc_auc_score(oos_labels, oos_scores),
        "auprc": average_precision_score(oos_labels, oos_scores),
    }


def ece(confidences: List[float], correct: List[int], n_bins: int = 15) -> float:
    """Expected Calibration Error。"""
    confidences = np.array(confidences)
    correct = np.array(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(confidences)) * abs(
                correct[mask].mean() - confidences[mask].mean()
            )
    return float(ece)


def coverage_jaccard(
    true: List[List[str]], pred: List[List[str]], tau: float = 0.5
) -> float:
    """集合级覆盖率：Jaccard ≥ tau 的比例。"""
    hits = sum(
        1 for t, p in zip(true, pred)
        if (len(set(t) & set(p)) / max(len(set(t) | set(p)), 1)) >= tau
    )
    return hits / max(len(true), 1)


def avg_set_size(pred: List[List[str]]) -> float:
    return float(np.mean([len(p) for p in pred])) if pred else 0.0
