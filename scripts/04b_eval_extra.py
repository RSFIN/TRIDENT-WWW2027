"""补充 L3 ECE 与 L5 误拒率计算（复用 04 的校准与决策管线）。

用法：
    python scripts/04b_eval_extra.py --dataset clinc150
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
from collections import Counter
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import macro_f1, ece, set_jaccard, coverage_jaccard, avg_set_size

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_fused_scores(records):
    all_fused = []
    for r in records:
        if r["oos"]:
            all_fused.append({})
            continue
        candidates = [(it, sim) for it, sim in r["retrieved"]]
        cand_ints = {it for it, _ in candidates}
        if r.get("llm_main") and r["llm_main"] not in cand_ints:
            candidates.append((r["llm_main"], 0.0))
        for a in (r.get("llm_aux") or []):
            existing = {it for it, _ in candidates}
            if a not in existing:
                candidates.append((a, 0.0))
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

    # === 校准（与 04 一致）===
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

    in_dom = [x for x in results if not x["record"]["oos"]]
    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"]["main"] for x in in_dom]

    # === L1 ===
    print(f"Main Macro-F1: {macro_f1(true_main, pred_main):.4f}")

    # === L3 ECE：用 top1 fused 分数作置信度（in-domain 主意图正确性）===
    confs, correct = [], []
    for r, fused in zip(test, test_fused):
        if r["oos"] or not fused:
            continue
        # 归一化 fused 分数到 [0,1]，越低越一致 → 置信度 = 1 - min
        top1_conf = 1.0 - min(fused.values())
        confs.append(top1_conf)
        correct.append(1 if r["main_intent"] in fused else 0)
    ece_val = ece(confs, correct, n_bins=15)
    print(f"ECE (15 bins, top1-fused): {ece_val:.4f}")

    # === L5 误拒率：in-domain 中被 reject 的比例 ===
    n_in = len(in_dom)
    n_reject_in = sum(1 for x in in_dom if x["decision"]["state"] == "reject")
    false_rej = n_reject_in / n_in if n_in else 0.0
    n_oos = sum(1 for x in results if x["record"]["oos"])
    n_reject_oos = sum(1 for x in results if x["record"]["oos"] and x["decision"]["state"] == "reject")
    print(f"in-domain: {n_in}, rejected in-domain: {n_reject_in}")
    print(f"OOS: {n_oos}, rejected OOS: {n_reject_oos}")
    print(f"误拒率 (in-domain reject / total in-domain): {false_rej:.4f}")
    print(f"OOS 拒识率 (OOS reject / total OOS): {n_reject_oos / n_oos if n_oos else 0:.4f}")

    # === 三态分布 ===
    states = Counter(x["decision"]["state"] for x in results)
    print(f"Decision dist: {dict(states)}")


if __name__ == "__main__":
    main()
