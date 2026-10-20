"""统计C2a消融基础分类指标：直接对比pred_main和真实标签。"""
import json
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent

datasets = ["clinc150", "mixatis", "mixsnips"]
results = {}

for ds in datasets:
    pred_dir = ROOT / "data" / "predictions" / f"{ds}_noex"
    test_file = pred_dir / "test.jsonl"
    if not test_file.exists():
        continue
    y_true = []
    y_pred = []
    n_oos = 0
    n_valid = 0
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["oos"]:
                n_oos +=1
                continue
            if not r["pred_main"]:
                continue
            y_true.append(r["main_intent"])
            y_pred.append(r["pred_main"])
            n_valid +=1
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    results[ds] = {
        "n_in_domain": n_valid,
        "n_oos": n_oos,
        "top1_acc": acc,
        "macro_f1": macro_f1,
    }
    print(f"\n=== {ds} (no examples ablation) ===")
    print(f"  In-domain samples: {n_valid}, OOS: {n_oos}")
    print(f"  Top-1 Accuracy: {acc:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")

# 保存
with open(ROOT / "data" / "predictions" / "c2a_basic_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n结果已保存至 c2a_basic_metrics.json")
