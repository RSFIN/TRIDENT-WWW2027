"""共形校准 + OOS gate + 三态决策 + L1-L5 指标评测。

用法：
    python scripts/04_calibrate_and_eval.py --dataset clinc150
"""
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import yaml
import numpy as np
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import (
    macro_f1, exact_match_set, set_jaccard, hit_at_k,
    oos_metrics, coverage_jaccard, avg_set_size,
)

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_fused_scores(records):
    """对每条预测记录算融合分数。"""
    all_fused = []
    for r in records:
        if r["oos"]:
            all_fused.append({})
            continue
        candidates = [(it, sim) for it, sim in r["retrieved"]]
        ret = f_ret(candidates)
        llm = f_llm(r, candidates)
        fused = fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"])
        all_fused.append(fused)
    return all_fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset

    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")

    # === 校准 ===
    calib_fused = build_fused_scores(calib)
    calib_main = [r["main_intent"] for r in calib if not r["oos"]]
    calib_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                  for r in calib if not r["oos"]]
    calib_fused_in = [f for f, r in zip(calib_fused, calib) if not r["oos"]]

    lam, meta = calibrate_threshold(
        calib_fused_in, calib_main, calib_aux,
        alpha=CFG["conformal"]["alpha"],
        tau_aux=CFG["conformal"]["tau_aux"],
        k_max=CFG["conformal"]["k_max"],
    )
    print(f"[calib] lambda={lam:.4f} empirical_risk={meta['empirical_risk']:.4f}")

    # OOS gate：in-domain 样本的 top1_sim 按 FPR95 标定
    in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
    theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])
    print(f"[oos] theta_oos={theta_oos:.4f} (FPR={CFG['conformal']['oos_fpr']})")

    # === 测试集推理 ===
    test_fused = build_fused_scores(test)
    results = []
    for r, fused in zip(test, test_fused):
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
            results.append({"record": r, "decision": d.__dict__})
            continue
        main, aux, sz = predict_set(fused, lam, CFG["conformal"]["k_max"])
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
        results.append({"record": r, "decision": d.__dict__})

    # === 指标 ===
    in_dom = [x for x in results if not x["record"]["oos"]]
    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"]["main"] for x in in_dom]
    true_aux = [[i for i in x["record"]["intents"] if i != x["record"]["main_intent"]]
                for x in in_dom]
    pred_aux = [x["decision"]["aux"] or [] for x in in_dom]
    pred_full = [x["decision"].get("candidates") or ([x["decision"]["main"]] if x["decision"]["main"] else [])
                 for x in in_dom]

    print("\n=== Results ===")
    print(f"Main Macro-F1:     {macro_f1(true_main, pred_main):.4f}")
    print(f"Set Exact Match:   {exact_match_set(true_aux, pred_aux):.4f}")
    print(f"Set Jaccard:       {set_jaccard(true_aux, pred_aux):.4f}")
    print(f"Coverage J>=0.5:   {coverage_jaccard(true_aux, pred_full, 0.5):.4f}")
    print(f"Avg set size:      {avg_set_size(pred_full):.2f}")

    # OOS
    oos_records = [x for x in results if x["record"]["oos"]]
    if oos_records:
        oos_labels = [1] * len(oos_records) + [0] * len(in_dom)
        oos_scores = [1 - x["record"]["top1_sim"] for x in oos_records] + \
                     [1 - x["record"]["top1_sim"] for x in in_dom]
        m = oos_metrics(oos_scores, oos_labels)
        print(f"OOS AUROC:         {m['auroc']:.4f}")
        print(f"OOS AUPRC:         {m['auprc']:.4f}")

    # 三态分布
    states = [x["decision"]["state"] for x in results]
    from collections import Counter
    print(f"Decision dist:     {Counter(states)}")


if __name__ == "__main__":
    main()
