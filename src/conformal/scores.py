"""非一致性分数：f_ret, f_LLM, rank 归一化, 融合。"""
from typing import Dict, List, Tuple
import numpy as np


def f_ret(candidates: List[Tuple[str, float]]) -> Dict[str, float]:
    """candidates: [(intent, cosine_sim)]。返回 {intent: 1-sim}，未召回 intent 给 +inf。"""
    return {intent: 1.0 - sim for intent, sim in candidates}


def f_llm(llm_result: Dict, candidates: List[Tuple[str, float]]) -> Dict[str, float]:
    """LLM 结构化输出 → 非一致性。

    利用 llm_main / llm_aux 字段构建分层信号：
      - llm_main: 非一致性 = 0.0（主意图，最强信号）
      - llm_aux:  非一致性 = 0.3
      - 其他候选: 非一致性 = 0.7
    confidence 数值仅在同层内做微调。
    """
    main = llm_result.get("llm_main", "")
    aux = set(llm_result.get("llm_aux") or [])
    out = {}
    for intent, _ in candidates:
        if intent == main:
            out[intent] = 0.0
        elif intent in aux:
            out[intent] = 0.3
        else:
            out[intent] = 0.7
    return out


def rank_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """在候选集内做 rank 归一化到 [0,1]。分越高越非一致。

    注意：按 (score, intent) 排序而不是只按 score。候选里经常出现分数完全相同的
    元素（f_llm 只有 0.0/0.3/0.7 三档），若只按 score 排序，Python 的稳定排序会
    保留输入顺序，而输入来自 set，其迭代顺序由 str 哈希决定、每个进程随机化。
    结果是同一份数据在不同进程里得到不同的预测集主意图选择，指标随之抖动。
    加上 intent 作为次级键即可得到全局确定的顺序。
    """
    items = sorted(scores.items(), key=lambda x: (x[1], x[0]))
    n = len(items)
    if n <= 1:
        return {k: 0.0 for k in scores}
    return {k: i / (n - 1) for i, (k, _) in enumerate(items)}


def fuse(
    ret_scores: Dict[str, float],
    llm_scores: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """融合：alpha * rank(f_ret) + (1-alpha) * rank(f_LLM)。"""
    all_intents = set(ret_scores) | set(llm_scores)
    # 补 +inf / 1.0 缺失
    ret_full = {i: ret_scores.get(i, float("inf")) for i in all_intents}
    llm_full = {i: llm_scores.get(i, 1.0) for i in all_intents}

    ret_rank = rank_normalize(ret_full)
    llm_rank = rank_normalize(llm_full)

    fused = {}
    for i in all_intents:
        if ret_full[i] == float("inf"):
            fused[i] = 1.0  # 未被检索召回，直接高非一致
        else:
            fused[i] = alpha * ret_rank[i] + (1 - alpha) * llm_rank[i]
    return fused
