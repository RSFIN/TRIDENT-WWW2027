"""C3: 配对bootstrap显著性检验 - TRIDENT指标的95%置信区间"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))

from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import macro_f1, set_jaccard, avg_set_size, ece


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_fused_scores(records, kr=None):
    if kr is None:
        kr = CFG["conformal"].get("k_r_use", None)
    all_fused = []
    for r in records:
        candidates = [(it, sim) for it, sim in r["retrieved"][:kr]] if kr else [(it, sim) for it, sim in r["retrieved"]]
        cand_ints = {it for it, _ in candidates}
        if r.get("llm_main") and r["llm_main"] not in cand_ints:
            candidates.append((r["llm_main"], 0.0))
        for a in (r.get("llm_aux") or []):
            existing = {it for it, _ in candidates}
            if a not in existing:
                candidates.append((a, 0.0))
        ret = f_ret(candidates)
        llm = f_llm(r, candidates)
        all_fused.append(fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"]))
    return all_fused


def compute_metrics(records, fused_scores, lam, theta_oos):
    """计算per-query的指标，返回每个样本的贡献"""
    per_query_iou = []
    per_query_correct = []
    per_query_set_size = []
    per_query_conf = []

    for r, fused in zip(records, fused_scores):
        main, aux, sz = predict_set(fused, lam, CFG["conformal"]["k_max"])
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])

        if d.state == "reject":
            pred_set = []
            per_query_iou.append(0.0)
            per_query_correct.append(0)
            per_query_set_size.append(0)
            per_query_conf.append(0.0)
        else:
            pred_set = d.candidates or ([d.main] + (d.aux or []))
            true_set = set(r["intents"])
            pred_set_s = set(pred_set)
            intersection = len(true_set & pred_set_s)
            union = len(true_set | pred_set_s)
            iou = intersection / union if union > 0 else 0.0
            per_query_iou.append(iou)
            per_query_correct.append(1 if d.main == r["main_intent"] else 0)
            per_query_set_size.append(len(pred_set))
            conf = 1.0 - fused.get(d.main, 1.0)
            per_query_conf.append(conf)

    return {
        "iou": np.array(per_query_iou),
        "correct": np.array(per_query_correct),
        "set_size": np.array(per_query_set_size),
        "conf": np.array(per_query_conf),
    }


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """计算bootstrap 95%置信区间"""
    n = len(data)
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(data[idx])
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper


def main():
    np.random.seed(42)
    datasets = ["clinc150", "mixatis", "mixsnips"]

    print("=" * 70)
    print("C3: Bootstrap 95%置信区间分析")
    print("=" * 70)

    for ds in datasets:
        calib_path = ROOT / "data" / "predictions" / ds / "calib.jsonl"
        test_path = ROOT / "data" / "predictions" / ds / "test.jsonl"

        if not calib_path.exists() or not test_path.exists():
            continue

        calib = load_pred(calib_path)
        test = load_pred(test_path)

        # 校准
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
        in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
        theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

        # 测试集in-domain样本
        test_ind = [r for r in test if not r["oos"]]
        test_fused_ind = build_fused_scores(test_ind)

        # 计算per-query指标
        per_query = compute_metrics(test_ind, test_fused_ind, lam, theta_oos)

        print(f"\n--- {ds.upper()} (n={len(test_ind)}) ---")
        print(f"{'Metric':<20} {'Mean':<10} {'95% CI':<20}")
        print("-" * 50)

        # IoU
        mean_iou, low_iou, high_iou = bootstrap_ci(per_query["iou"])
        print(f"{'Set IoU':<20} {mean_iou:<10.3f} [{low_iou:.3f}, {high_iou:.3f}]")

        # F1 (main intent)
        mean_f1, low_f1, high_f1 = bootstrap_ci(per_query["correct"])
        print(f"{'Main Intent Acc':<20} {mean_f1:<10.3f} [{low_f1:.3f}, {high_f1:.3f}]")

        # Avg set size
        mean_size, low_size, high_size = bootstrap_ci(per_query["set_size"])
        print(f"{'Avg Set Size':<20} {mean_size:<10.2f} [{low_size:.2f}, {high_size:.2f}]")

    print("\n" + "=" * 70)
    print("结论：")
    print("我们报告TRIDENT自身指标的bootstrap 95%置信区间")
    print("与基线的配对检验由于缺乏基线per-query结果，")
    print("使用两个 seed 的标准差来估计不确定性")
    print("=" * 70)


if __name__ == "__main__":
    main()
