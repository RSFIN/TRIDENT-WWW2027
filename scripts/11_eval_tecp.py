"""R11: TECP 评测——用采样 entropy 做 split conformal。

读 data/predictions/{dataset}/tecp_{split}.jsonl，
对每条算 main intent 预测分布熵，calib 上取分位数，test 上评测。

用法：
    python scripts/11_eval_tecp.py --dataset clinc150 --alpha 0.1
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def entropy_of(samples):
    """对 T 次采样的 main intent 列表算 Shannon entropy（自然对数）。"""
    if not samples: return 0.0
    cnt = Counter(samples)
    n = sum(cnt.values())
    p = np.array([c/n for c in cnt.values()])
    return float(-(p * np.log(p + 1e-12)).sum())


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    pred_dir = ROOT / "data" / "predictions" / args.dataset
    calib = load(pred_dir / "tecp_calib.jsonl")
    test = load(pred_dir / "tecp_test.jsonl")

    # calib: 算 entropy 和是否覆盖（argmax 是否 == main_intent）
    calib_scores, calib_errs = [], []
    for r in calib:
        if r["oos"]: continue
        H = entropy_of(r["samples"])
        pred_main = Counter(r["samples"]).most_common(1)[0][0] if r["samples"] else ""
        calib_scores.append(H)
        calib_errs.append(0 if pred_main == r["main_intent"] else 1)

    # split CP: 取分位数作为阈值（非conformity score = entropy；error=1 时 score=H，error=0 时 score=0？
    # TECP 思路：entropy 越高越不确定，直接卡 entropy 阈值。
    q = np.quantile(calib_scores, 1 - args.alpha)
    print(f"[tecp] threshold H* = {q:.4f} (alpha={args.alpha})")

    # test: 评测
    n = len(test); correct = 0; rejected = 0; oos_auroc_scores = []; oos_labels = []
    for r in test:
        H = entropy_of(r["samples"])
        pred_main = Counter(r["samples"]).most_common(1)[0][0] if r["samples"] else ""
        # OOS: entropy 越高越可能 OOS
        oos_auroc_scores.append(H); oos_labels.append(1 if r["oos"] else 0)
        if H >= q:
            rejected += 1  # abstain
        else:
            if pred_main == r["main_intent"]: correct += 1

    answered = n - rejected
    acc = correct / answered if answered else 0
    print(f"=== TECP results ({args.dataset}) ===")
    print(f"Answered: {answered}/{n} ({answered/n:.1%}), Rejected: {rejected}")
    print(f"Acc on answered: {acc:.4f}")
    # OOS AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(oos_labels, oos_auroc_scores)
    print(f"OOS AUROC (entropy score): {auroc:.4f}")


if __name__ == "__main__":
    main()
