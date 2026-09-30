"""A2 温度缩放校准对比：LLM 置信度温度缩放 + 标准 split CP（单标签，CICC 风格）。

分数：对每条样本的主意图候选，LLM confidence c → logit = log(c/(1-c))，
      p_T = sigmoid(logit / T)；nonconformity s = -log p_T（越高越不确信）。
校准：calib 上对 T ∈ 网格，取主意图 s 的 split CP 分位数 λ_T；
      选 coverage ≥ 1-α 且 ECE 最小的 T（保持 CP 保证的前提下最优校准）。
推理：预测集 P = {i: s_T(i) ≤ λ_T}（单标签，无 aux 扩展），叠加 OOS gate + 三态决策。
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
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import (
    macro_f1, exact_match_set, set_jaccard, hit_at_k,
    oos_metrics, coverage_jaccard, avg_set_size, ece,
)

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
ALPHA = CFG["conformal"]["alpha"]
T_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def temp_score(r, T):
    """主意图非一致性分数：-log p_T，基于 LLM confidence。"""
    conf = r.get("llm_confidence") or {}
    c = conf.get(r.get("llm_main", ""), 0.0)
    c = min(max(c, 1e-6), 1.0 - 1e-6)
    logit = np.log(c / (1.0 - c))
    p_T = 1.0 / (1.0 + np.exp(-logit / T))
    return -np.log(max(p_T, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset

    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")
    k_max = CFG["conformal"]["k_max"]

    # 校准样本（in-domain）
    calib_in = [r for r in calib if not r["oos"]]
    in_dom_sims = [r["top1_sim"] for r in calib_in]
    theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

    # 对每个 T：split CP 阈值 + calib coverage + ECE
    best = None
    rows = []
    for T in T_GRID:
        s_calib = np.array(sorted(temp_score(r, T) for r in calib_in))
        n = len(s_calib)
        q_idx = int(np.ceil((n + 1) * (1.0 - ALPHA))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        lam = s_calib[q_idx]
        cov = np.mean(s_calib <= lam)
        # calib ECE（预测集主意图）
        confs, corrects = [], []
        for r in calib_in:
            s = temp_score(r, T)
            if s <= lam:
                confs.append(1.0 - np.exp(-s))
                corrects.append(1 if r["llm_main"] == r["main_intent"] else 0)
            else:
                confs.append(0.0)
                corrects.append(0)
        ece_cal = ece(confs, corrects, 15)
        rows.append((T, lam, cov, ece_cal))
        if cov >= 1.0 - ALPHA and (best is None or ece_cal < best[3]):
            best = (T, lam, cov, ece_cal)

    if best is None:
        best = min(rows, key=lambda x: x[2])  # 兜底：coverage 最高的 T
    T, lam, cov_cal, ece_cal = best
    print(f"[A2 {ds}] 温度缩放：T={T:.2f} lam={lam:.4f} calib_cov={cov_cal:.4f} calib_ECE={ece_cal:.4f}")
    for T_, lam_, cov_, ece_ in rows:
        print(f"  T={T_:.2f}: lam={lam_:.4f} cov={cov_:.4f} ece={ece_:.4f}")

    # 测试推理（单标签预测集 + OOS gate）
    results = []
    for r in test:
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, k_max)
            results.append({"record": r, "decision": d.__dict__})
            continue
        s_main = temp_score(r, T)
        if s_main <= lam:
            main = r.get("llm_main", "")
            d = decide(main, [], 1, r["top1_sim"], theta_oos, k_max)
        else:
            d = decide("", [], 0, r["top1_sim"], theta_oos, k_max)
        results.append({"record": r, "decision": d.__dict__})

    in_dom = [x for x in results if not x["record"]["oos"]]
    true_full = [x["record"]["intents"] for x in in_dom]
    pred_full = []
    for x in in_dom:
        d = x["decision"]
        if d["state"] == "reject":
            pred_full.append([])
        elif d["state"] == "release":
            pred_full.append([d["main"]])
        else:
            pred_full.append([d["main"]] + (d["aux"] or []))
    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"]["main"] for x in in_dom]

    f1 = macro_f1(true_main, pred_main)
    exact = exact_match_set(true_full, pred_full)
    iou = set_jaccard(true_full, pred_full)
    hit1 = hit_at_k(true_full, pred_full, 1)
    hit3 = hit_at_k(true_full, pred_full, 3)
    hit5 = hit_at_k(true_full, pred_full, 5)
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
            s = temp_score(x["record"], T)
            confs.append(max(0.0, 1.0 - np.exp(-s)))
            corrects.append(1 if d["main"] == x["record"]["main_intent"] else 0)
    ece_val = ece(confs, corrects, 15)

    n_in = len(in_dom)
    n_reject_in = sum(1 for x in in_dom if x["decision"]["state"] == "reject")
    n_clarify = sum(1 for x in in_dom if x["decision"]["state"] == "clarify")

    oos_records = [x for x in results if x["record"]["oos"]]
    if oos_records:
        oos_rej = sum(1 for x in oos_records if x["decision"]["state"] == "reject") / len(oos_records)
        oos_labels = [1] * len(oos_records) + [0] * n_in
        oos_scores = [1 - x["record"]["top1_sim"] for x in oos_records] + \
                     [1 - x["record"]["top1_sim"] for x in in_dom]
        m2 = oos_metrics(oos_scores, oos_labels)
    else:
        oos_rej, m2 = None, None

    tag = f"ablation_temp_{ds}"
    lines = [
        f"[A2 {tag}] T={T:.2f} lam={lam:.4f} calib_cov={cov_cal:.4f} calib_ECE={ece_cal:.4f}",
        f"Main Macro-F1:      {f1:.4f}",
        f"Set Exact Match:    {exact:.4f}",
        f"Set IoU:            {iou:.4f}",
        f"Hit@1/3/5:          {hit1:.4f} / {hit3:.4f} / {hit5:.4f}",
        f"Coverage(J>=0.5):   {cov05:.4f}",
        f"Coverage(含):       {cov_any:.4f}",
        f"Avg set size:       {asize:.2f}",
        f"ECE (15 bins):      {ece_val:.4f}",
        f"误拒率:             {n_reject_in / n_in:.4f} ({n_reject_in}/{n_in})",
        f"澄清触发率:         {n_clarify / n_in:.4f}",
    ]
    if oos_records:
        lines += [
            f"OOS 拒识率:         {oos_rej:.4f}",
            f"OOS AUROC/AUPRC:    {m2['auroc']:.4f} / {m2['auprc']:.4f}",
        ]
    lines.append(f"Decision dist:      {dict(Counter(x['decision']['state'] for x in results))}")

    text = "\n".join(lines)
    print(text)
    with open(ROOT / "results" / f"{tag}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
