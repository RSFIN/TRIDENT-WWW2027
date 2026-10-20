"""C5: τ_hi tie-breaking重报 - 完全盲测OOS拒识率（不用ground truth）"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))

from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_fused_scores_blind(records, kr=None):
    """盲测：不管是不是OOS，都正常计算fused分数"""
    if kr is None:
        kr = CFG["conformal"].get("k_r_use", None)
    all_fused = []
    for r in records:
        # 不管是不是OOS，都正常计算
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


def main():
    datasets = ["clinc150"]

    print("=" * 70)
    print("C5: 完全盲测OOS拒识率（不使用ground truth OOS标签）")
    print("=" * 70)

    for ds in datasets:
        calib_path = ROOT / "data" / "predictions" / ds / "calib.jsonl"
        test_path = ROOT / "data" / "predictions" / ds / "test.jsonl"

        calib = load_pred(calib_path)
        test = load_pred(test_path)

        # 校准：只用in-domain样本校准
        calib_ind = [r for r in calib if not r["oos"]]
        calib_fused_ind = build_fused_scores_blind(calib_ind)
        calib_main = [r["main_intent"] for r in calib_ind]
        calib_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                     for r in calib_ind]
        lam, meta = calibrate_threshold(
            calib_fused_ind, calib_main, calib_aux,
            alpha=CFG["conformal"]["alpha"],
            tau_aux=CFG["conformal"]["tau_aux"],
            k_max=CFG["conformal"]["k_max"],
        )
        in_dom_sims = [r["top1_sim"] for r in calib_ind]
        theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

        print(f"\n--- {ds.upper()} ---")
        print(f"lambda={lam:.4f}, theta_oos={theta_oos:.4f}")

        # 盲测推理：对所有样本（包括OOS）都正常计算分数
        test_fused = build_fused_scores_blind(test)
        results = []
        for r, fused in zip(test, test_fused):
            # 不管是不是OOS，都正常predict_set
            main, aux, sz = predict_set(fused, lam, CFG["conformal"]["k_max"])
            # decide函数里会用top1_sim < theta_oos判断OOS
            d = decide(main, aux, sz, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
            results.append({"record": r, "decision": d, "fused": fused})

        # 计算指标
        oos_records = [x for x in results if x["record"]["oos"]]
        in_dom_records = [x for x in results if not x["record"]["oos"]]

        oos_rej = sum(1 for x in oos_records if x["decision"].state == "reject") / len(oos_records)
        in_dom_rej = sum(1 for x in in_dom_records if x["decision"].state == "reject") / len(in_dom_records)

        print(f"\n盲测OOS拒识率: {oos_rej:.4f} ({sum(1 for x in oos_records if x['decision'].state == 'reject')}/{len(oos_records)})")
        print(f"In-domain误拒率: {in_dom_rej:.4f}")

        # 看看OOS样本的decision分布
        from collections import Counter
        oos_states = Counter(x["decision"].state for x in oos_records)
        print(f"\nOOS样本decision分布: {dict(oos_states)}")

        # 看看有多少OOS样本被release了（错误放行）
        oos_released = [x for x in oos_records if x["decision"].state == "release"]
        oos_clarify = [x for x in oos_records if x["decision"].state == "clarify"]
        print(f"\nOOS被错误release: {len(oos_released)}")
        print(f"OOS被错误clarify: {len(oos_clarify)}")
        if len(oos_released) > 0 or len(oos_clarify) > 0:
            wrong = oos_released + oos_clarify
            sims = [x["record"]["top1_sim"] for x in wrong]
            print(f"  错误放行的样本top1_sim范围: {min(sims):.4f} ~ {max(sims):.4f}")
            print(f"  theta_oos = {theta_oos:.4f}")
            print(f"  也就是说，有{len(wrong)}个OOS样本的top1_sim >= theta_oos，所以没被拒识")

    print("\n" + "=" * 70)
    print("结论：")
    print("之前的OOS拒识率100%是评测时用了ground truth OOS标签")
    print("真实盲测下，OOS拒识率应该是上面这个数字")
    print("=" * 70)


if __name__ == "__main__":
    main()
