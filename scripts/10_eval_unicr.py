"""R10: UniCR-style learned fusion 基线。

在 calib 集上拟合 logistic calibration head，融合四类信号预测"候选意图是否属于真实意图集"，
以 1 - p(correct) 作为非一致性分数，再做单标签 split CP（主意图上的分位数阈值），
叠加 OOS gate 与三态决策，输出统一口径 v1.0 全套指标（与 05_eval_unified.py 对齐）。

特征（每候选意图 i、样本 q）：
    x1 = f_ret(q,i)      = 1 - sim(q,i)            （未召回给 1.0）
    x2 = f_LLM(q,i)      = 分层非一致性（main=0, aux=0.3, other=0.7）
    x3 = 1 - top1_sim(q)                          （样本级检索不确定性，广播）
    x4 = 1 - llm_confidence(q,i)                  （LLM 自评置信度反向）

训练标签：y = 1[i ∈ intents(q)]（多意图可多标签为 1）。
校准：calib 主意图的非一致性分数取 (n+1)(1-α)/n 分位数作为阈值。
推理：预测集 P = {i: s(q,i) ≤ θ}，reject 空集；OOS gate 用 top1_sim FPR95 标定。

用法：
    python scripts/10_eval_unicr.py --dataset clinc150
    python scripts/10_eval_unicr.py --dataset mixatis
    python scripts/10_eval_unicr.py --dataset mixsnips
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.conformal.scores import f_llm
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import (
    macro_f1, exact_match_set, set_jaccard, hit_at_k,
    oos_metrics, coverage_jaccard, avg_set_size, ece,
)

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
ALPHA = CFG["conformal"]["alpha"]


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def candidate_set(r):
    """候选意图空间 = retrieved + llm_main + llm_aux（与 04/05 的 build_fused_scores 一致）。"""
    cands = [(it, sim) for it, sim in r["retrieved"]]
    cand_ints = {it for it, _ in cands}
    if r.get("llm_main") and r["llm_main"] not in cand_ints:
        cands.append((r["llm_main"], 0.0))
    for a in (r.get("llm_aux") or []):
        existing = {it for it, _ in cands}
        if a not in existing:
            cands.append((a, 0.0))
    return cands


def feature_row(r, intent, sim):
    """四维特征，方向为"越非一致越大"。"""
    llm_sc = f_llm(r, [(intent, sim)])
    x2 = llm_sc.get(intent, 0.7)
    x4 = 1.0 - (r.get("llm_confidence") or {}).get(intent, 0.0)
    return [1.0 - sim, x2, 1.0 - r["top1_sim"], x4]


def fit_learned_score(calib):
    """在 calib（in-domain）上训练 logistic head，返回 score 函数。"""
    X, y = [], []
    for r in calib:
        if r["oos"]:
            continue
        true_set = set(r["intents"])
        for intent, sim in candidate_set(r):
            X.append(feature_row(r, intent, sim))
            y.append(1 if intent in true_set else 0)
    X = np.array(X, dtype=float)
    y = np.array(y)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(scaler.transform(X), y)

    def score(r):
        # 用 decision function 的负值：决策值越大越可能正确 → 分数越小越可能正确
        out = {}
        for intent, sim in candidate_set(r):
            x = scaler.transform([feature_row(r, intent, sim)])
            out[intent] = -float(clf.decision_function(x)[0])
        return out

    def proba(r):
        # logistic 概率：作为 ECE 置信度来源（语义与 Ours 的 fused 置信度对应）
        out = {}
        for intent, sim in candidate_set(r):
            x = scaler.transform([feature_row(r, intent, sim)])
            out[intent] = float(clf.predict_proba(x)[0, 1])
        return out

    return score, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset

    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")

    # === 训练 learned score ===
    score, proba = fit_learned_score(calib)

    # === 校准：主意图 split CP 分位数阈值 ===
    calib_scores = []
    for r in calib:
        if r["oos"]:
            continue
        s = score(r)
        calib_scores.append(s.get(r["main_intent"], 1.0))
    calib_scores = np.array(sorted(calib_scores))
    n_cal = len(calib_scores)
    # 标准 split CP：θ = 排序后第 ceil((n+1)(1-α)) 个（0-based 索引再 -1）
    q_idx = int(np.ceil((n_cal + 1) * (1.0 - ALPHA))) - 1
    q_idx = max(0, min(q_idx, n_cal - 1))
    theta = calib_scores[q_idx]
    emp_cov = np.mean(calib_scores <= theta)

    # OOS gate：与 Ours 相同，top1_sim FPR95
    in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
    theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

    # === 测试推理 ===
    results = []
    for r in test:
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
            results.append({"record": r, "decision": d.__dict__, "scores": {}, "proba": {}})
            continue
        s = score(r)
        p = proba(r)
        pred = {i: v for i, v in s.items() if v <= theta}
        if not pred:
            d = decide("", [], 0, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
            results.append({"record": r, "decision": d.__dict__, "scores": s, "proba": p})
            continue
        sorted_ints = sorted(pred.items(), key=lambda x: x[1])
        main = sorted_ints[0][0]
        aux = [i for i, _ in sorted_ints[1:CFG["conformal"]["k_max"] + 1]]
        d = decide(main, aux, len(sorted_ints), r["top1_sim"], theta_oos,
                   CFG["conformal"]["k_max"])
        results.append({"record": r, "decision": d.__dict__, "scores": s, "proba": p})

    # === 统一口径指标 ===
    in_dom = [x for x in results if not x["record"]["oos"]]
    true_full = [x["record"]["intents"] for x in in_dom]
    pred_full, pred_ranked = [], []
    for x in in_dom:
        d = x["decision"]
        if d["state"] == "reject":
            pred_full.append([])
            pred_ranked.append([])
        elif d["state"] == "release":
            pred_full.append([d["main"]])
            pred_ranked.append([d["main"]])
        else:
            cands = d.get("candidates") or ([d["main"]] + (d["aux"] or []))
            s = x["scores"]
            ordered = sorted(cands, key=lambda i: s.get(i, 1.0))
            pred_full.append(ordered)
            pred_ranked.append(ordered)

    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"]["main"] for x in in_dom]

    f1 = macro_f1(true_main, pred_main)
    exact = exact_match_set(true_full, pred_full)
    iou = set_jaccard(true_full, pred_full)
    hit1 = hit_at_k(true_full, pred_ranked, 1)
    hit3 = hit_at_k(true_full, pred_ranked, 3)
    hit5 = hit_at_k(true_full, pred_ranked, 5)
    cov05 = coverage_jaccard(true_full, pred_full, 0.5)
    cov_any = sum(1 for t, p in zip(true_full, pred_full) if set(t) & set(p)) / max(len(in_dom), 1)
    asize = avg_set_size(pred_full)

    confs, corrects = [], []
    for x in in_dom:
        d = x["decision"]
        if d["state"] == "reject":
            confs.append(0.0)
            corrects.append(0)
        else:
            p = x["proba"]
            conf = p.get(d["main"], 0.0)
            confs.append(conf)
            corrects.append(1 if d["main"] == x["record"]["main_intent"] else 0)
    ece_val = ece(confs, corrects, 15)

    n_in = len(in_dom)
    n_reject_in = sum(1 for x in in_dom if x["decision"]["state"] == "reject")
    n_clarify = sum(1 for x in in_dom if x["decision"]["state"] == "clarify")
    frr = n_reject_in / n_in
    clarify_rate = n_clarify / n_in

    oos_records = [x for x in results if x["record"]["oos"]]
    if oos_records:
        oos_rej = sum(1 for x in oos_records if x["decision"]["state"] == "reject") / len(oos_records)
        oos_labels = [1] * len(oos_records) + [0] * n_in
        oos_scores = [1 - x["record"]["top1_sim"] for x in oos_records] + \
                     [1 - x["record"]["top1_sim"] for x in in_dom]
        m2 = oos_metrics(oos_scores, oos_labels)
    else:
        oos_rej, m2 = None, None

    # === 输出 ===
    print(f"[R10 {ds}] UniCR-style learned fusion, alpha={ALPHA}, full-level 统一口径")
    print(f"theta={theta:.4f} emp_cov={emp_cov:.4f} (n_cal={n_cal}) theta_oos={theta_oos:.4f}")
    print(f"Main Macro-F1:      {f1:.4f}")
    print(f"Set Exact Match:    {exact:.4f}")
    print(f"Set IoU:            {iou:.4f}")
    print(f"Hit@1/3/5:          {hit1:.4f} / {hit3:.4f} / {hit5:.4f}")
    print(f"Coverage(J>=0.5):   {cov05:.4f}")
    print(f"Coverage(含):       {cov_any:.4f}")
    print(f"Avg set size:       {asize:.2f}")
    print(f"ECE (15 bins):      {ece_val:.4f}")
    print(f"误拒率:             {frr:.4f} ({n_reject_in}/{n_in})")
    print(f"澄清触发率:         {clarify_rate:.4f}")
    if oos_records:
        print(f"OOS 拒识率:         {oos_rej:.4f}")
        print(f"OOS AUROC/AUPRC:    {m2['auroc']:.4f} / {m2['auprc']:.4f}")
    print(f"Decision dist:      {dict(Counter(x['decision']['state'] for x in results))}")


if __name__ == "__main__":
    main()
