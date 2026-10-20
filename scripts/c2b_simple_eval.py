"""统计C2b消融基础分类指标：few-shot基线补等量检索evidence。"""
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent

datasets = ["clinc150", "mixatis", "mixsnips"]
results = {}

for ds in datasets:
    pred_dir = ROOT / "data" / "predictions" / f"{ds}_fewshot_ev"
    test_file = pred_dir / "test.jsonl"
    if not test_file.exists():
        print(f"[WARN] {ds} test not found")
        continue
    y_true = []
    y_pred = []
    n_valid = 0
    n_fail = 0
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("pred_main"):
                n_fail += 1
                continue
            y_true.append(r["label"])
            y_pred.append(r["pred_main"])
            n_valid += 1
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results[ds] = {
        "n_valid": n_valid,
        "n_fail": n_fail,
        "top1_acc": acc,
        "macro_f1": macro_f1,
    }
    print(f"\n=== {ds} (few-shot + evidence) ===")
    print(f"  Valid samples: {n_valid}, Fail: {n_fail}")
    print(f"  Top-1 Accuracy: {acc:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")

with open(ROOT / "data" / "predictions" / "c2b_basic_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n结果已保存至 c2b_basic_metrics.json")
