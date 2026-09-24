"""三态决策 + OOS dual threshold。"""
from dataclasses import dataclass
from typing import List


@dataclass
class Decision:
    state: str          # "release" | "clarify" | "reject"
    main: str = ""
    aux: List[str] = None
    candidates: List[str] = None   # 澄清时返回的候选集
    set_size: int = 0


def decide(
    main: str,
    aux: List[str],
    set_size: int,
    top1_sim: float,
    theta_oos: float,
    k_max: int = 3,
) -> Decision:
    """三态：放行 / 澄清 / 拒识。

    top1_sim < theta_oos → OOS 拒识（dual threshold）
    set_size == 1 → 放行
    1 < set_size <= k_max → 澄清
    set_size == 0 → 拒识
    """
    if set_size == 0 or top1_sim < theta_oos:
        return Decision(state="reject", set_size=set_size)
    if set_size == 1:
        return Decision(state="release", main=main, aux=aux, set_size=set_size)
    # 2..k_max
    return Decision(
        state="clarify", main=main, aux=aux,
        candidates=[main] + aux, set_size=set_size,
    )


def calibrate_oos_threshold(
    in_domain_sims: List[float],
    target_fpr: float = 0.05,
) -> float:
    """按 FPR95 标定 OOS gate：95% in-domain 通过。"""
    import numpy as np
    return float(np.quantile(in_domain_sims, target_fpr))
