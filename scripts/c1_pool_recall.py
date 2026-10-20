"""C1: 候选池召回率分析 - 真实意图落入top-K_r检索池的比例"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def compute_pool_recall(records, kr_values=[1, 3, 5, 10]):
    """计算每个K_r下，真实意图集合至少有一个落入候选池的比例"""
    in_domain = [r for r in records if not r["oos"]]
    total = len(in_domain)

    results = {}
    for kr in kr_values:
        # 真实意图至少有一个在top-kr里的比例（单召回）
        hit_any = 0
        # 真实意图全部在top-kr里的比例（全召回）
        hit_all = 0
        # 平均覆盖比例
        avg_coverage = 0.0

        for r in in_domain:
            retrieved = [it for it, _ in r["retrieved"][:kr]]
            true_intents = set(r["intents"])
            retrieved_set = set(retrieved)

            intersection = true_intents & retrieved_set
            if len(intersection) > 0:
                hit_any += 1
            if intersection == true_intents:
                hit_all += 1
            avg_coverage += len(intersection) / len(true_intents)

        results[kr] = {
            "recall_any": hit_any / total,  # 至少一个命中
            "recall_all": hit_all / total,  # 全部命中
            "avg_jaccard_coverage": avg_coverage / total,  # 平均覆盖比例
            "total": total,
        }
    return results


def main():
    datasets = ["clinc150", "mixatis", "mixsnips"]
    kr_values = [1, 3, 5, 10]

    print("=" * 70)
    print("C1: 候选池召回率分析 (真实意图落入top-K_r检索池的比例)")
    print("=" * 70)

    for ds in datasets:
        test_path = ROOT / "data" / "predictions" / ds / "test.jsonl"
        if not test_path.exists():
            print(f"\n{ds}: 测试文件不存在，跳过")
            continue

        test = load_pred(test_path)
        results = compute_pool_recall(test, kr_values)

        print(f"\n--- {ds.upper()} ---")
        print(f"{'K_r':<6} {'Recall@any':<15} {'Recall@all':<15} {'AvgCoverage':<15}")
        print("-" * 55)
        for kr in kr_values:
            r = results[kr]
            print(f"{kr:<6} {r['recall_any']:<15.3f} {r['recall_all']:<15.3f} {r['avg_jaccard_coverage']:<15.3f}")

        # 额外分析：多意图样本的召回率
        multi_intent = [r for r in test if not r["oos"] and len(r["intents"]) > 1]
        if len(multi_intent) > 0:
            print(f"\n  多意图样本数: {len(multi_intent)}")
            multi_results = compute_pool_recall(multi_intent, kr_values)
            for kr in kr_values:
                r = multi_results[kr]
                print(f"  K_r={kr}: multi-intent recall_all = {r['recall_all']:.3f}")

    print("\n" + "=" * 70)
    print("结论：K_r=1时，MixATIS/MixSNIPS的多意图样本召回率很低，")
    print("这解释了为什么MixATIS无法满足α=0.1的风险约束。")
    print("=" * 70)


if __name__ == "__main__":
    main()
