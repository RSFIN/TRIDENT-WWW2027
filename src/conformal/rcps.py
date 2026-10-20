"""Risk-Controlling Prediction Sets (Bates et al. 2021) 简化实现。

分层风险度量：
    L(Ĉ, Y) = 1[main 错] + 1[main 对 且 aux Jaccard < tau']

校准流程：
    1. 对每个校准样本，计算融合分数 s(q, i)
    2. 对每个候选阈值 λ，模拟预测集 Ĉ_λ(q) = {i: s(q,i) ≤ λ}
    3. 计算经验风险 R̂(λ) = mean L(Ĉ_λ, Y)
    4. 取最小的 λ 使 R̂(λ) ≤ α
"""
from typing import Dict, List, Tuple
import numpy as np


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def stratified_loss(
    pred_main: str,
    pred_aux: List[str],
    true_main: str,
    true_aux: List[str],
    tau_aux: float = 0.5,
) -> float:
    """分层 loss，返回 {0, 0.3, 0.5, 1.0}。

    设计原则：正确答案在预测集合里就算部分正确。
      - 0.0: main 对 且 aux Jaccard >= tau（集合完全覆盖）
      - 0.3: main 对 但 aux Jaccard < tau（默认对但次意图不全）
      - 0.5: main 错 但 true_main 在 pred_aux 里（给了候选但默认错了）
      - 1.0: true_main 完全不在预测集合里
    """
    pred_set = {pred_main} | set(pred_aux)
    main_ok = (pred_main == true_main)
    main_covered = true_main in pred_set

    if not main_covered:
        return 1.0
    if main_ok:
        aux_j = jaccard(pred_aux, true_aux)
        if aux_j >= tau_aux:
            return 0.0
        return 0.3
    return 0.5  # main 错但 true_main 在 aux 里


def calibrate_threshold(
    calib_scores: List[Dict[str, float]],   # 每样本 {intent: fused_score}
    calib_main: List[str],
    calib_aux: List[List[str]],
    alpha: float = 0.1,
    tau_aux: float = 0.5,
    k_max: int = 3,
) -> Tuple[float, Dict]:
    """在校准集上选最小 λ 使经验风险 ≤ α。

    返回 (lambda_hat, meta)。若整条风险曲线都无法降到 α 以下（目标不可达），
    返回 λ=1.0（最大预测集）及其实际经验风险，并在 meta["feasible"]=False 标记，
    而不是返回 1.0 这个哨兵值——否则下游会把"没有可行解"误读成"风险 = 100%"。
    """
    # 收集所有候选分数作为阈值候选
    all_scores = sorted({s for d in calib_scores for s in d.values() if s < float("inf")})
    # 加边界
    candidates = [0.0] + all_scores + [1.0]

    best_lam = 1.0
    best_risk = 1.0
    feasible = False
    risk_at_lam1 = 1.0     # λ=1.0 时的实际风险，用于不可达时的报告
    for lam in candidates:
        losses = []
        for d, true_m, true_a in zip(calib_scores, calib_main, calib_aux):
            pred = {i: s for i, s in d.items() if s <= lam and s < float("inf")}
            if not pred:
                # 空集 = 拒识，对 in-domain 样本算 loss
                losses.append(1.0)
                continue
            # 分数最低的为主意图，其余按分数升序为 aux
            sorted_intents = sorted(pred.items(), key=lambda x: x[1])
            pred_main = sorted_intents[0][0]
            pred_aux = [i for i, _ in sorted_intents[1:k_max + 1]]
            losses.append(stratified_loss(pred_main, pred_aux, true_m, true_a, tau_aux))
        risk = float(np.mean(losses))
        if lam >= 1.0:
            risk_at_lam1 = risk
        if risk <= alpha and not feasible:
            best_lam = lam
            best_risk = risk
            feasible = True
            break  # 从小到大，第一个满足的就是最小 λ

    if not feasible:
        best_lam = 1.0
        best_risk = risk_at_lam1

    meta = {
        "empirical_risk": best_risk,
        "n_calib": len(calib_scores),
        "feasible": feasible,
        "tau_aux": tau_aux,
        "k_max": k_max,
    }
    return best_lam, meta


def predict_set(
    fused_scores: Dict[str, float],
    lam: float,
    k_max: int = 3,
) -> Tuple[str, List[str], int]:
    """给定融合分数和 λ，返回 (main, aux, set_size)。空集 = 拒识。"""
    pred = {i: s for i, s in fused_scores.items() if s <= lam and s < float("inf")}
    if not pred:
        return "", [], 0
    sorted_intents = sorted(pred.items(), key=lambda x: x[1])
    main = sorted_intents[0][0]
    aux = [i for i, _ in sorted_intents[1:k_max + 1]]
    return main, aux, len(sorted_intents)
