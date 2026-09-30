"""统一口径评估 v1.0：full 级集合指标 + Hit@k + Cov(含) + ECE + 误拒率 + 澄清率。

对照 docs/统一指标口径规范_v1.md：
  - 集合指标全部在完整意图集合上计算（true = r["intents"]，pred = 完整候选集）
  - reject → P = ∅，计入所有指标分母
  - ECE = 主意图 top1-fused 置信度（1 − min fused score），15 bins
  - 误拒率 = in-domain 被判 reject 比例；OOS 拒识率仅 CLINC150

用法：
    python scripts/05_eval_unified.py --dataset clinc150
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

import yaml
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import (
    macro_f1, exact_match_set, set_jaccard, hit_at_k,
    oos_metrics, coverage_jaccard, avg_set_size, ece,
)

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))


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
        if r["oos"]:
            all_fused.append({})
            continue
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset

    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")

    # === 校准（同 04） ===
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

    # === 测试推理 ===
    test_fused = build_fused_scores(test)
    results = []
    for r, fused in zip(test, test_fused):
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
            results.append({"record": r, "decision": d, "fused": fused})
            continue
        main, aux, sz = predict_set(fused, lam, CFG["conformal"]["k_max"])
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, CFG["conformal"]["k_max"])
        results.append({"record": r, "decision": d, "fused": fused})

    # === 统一口径指标 ===
    in_dom = [x for x in results if not x["record"]["oos"]]
    true_full = [x["record"]["intents"] for x in in_dom]
    pred_full, pred_ranked = [], []
    for x in in_dom:
        d = x["decision"]
        if d.state == "reject":
            pred_full.append([])
            pred_ranked.append([])
        elif d.state == "release":
            pred_full.append([d.main])
            pred_ranked.append([d.main])
        else:  # clarify
            cands = d.candidates or ([d.main] + (d.aux or []))
            fused = x["fused"]
            ordered = sorted(cands, key=lambda i: fused.get(i, 1.0))
            pred_full.append(ordered)
            pred_ranked.append(ordered)

    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"].main for x in in_dom]

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
        if d.state == "reject":
            confs.append(0.0)
            corrects.append(0)
        else:
            fused = x["fused"]
            conf = 1.0 - fused.get(d.main, 1.0)
            confs.append(conf)
            corrects.append(1 if d.main == x["record"]["main_intent"] else 0)
    ece_val = ece(confs, corrects, 15)

    n_in = len(in_dom)
    n_reject_in = sum(1 for x in in_dom if x["decision"].state == "reject")
    n_clarify = sum(1 for x in in_dom if x["decision"].state == "clarify")
    frr = n_reject_in / n_in
    clarify_rate = n_clarify / n_in

    oos_records = [x for x in results if x["record"]["oos"]]
    if oos_records:
        oos_rej = sum(1 for x in oos_records if x["decision"].state == "reject") / len(oos_records)
        oos_labels = [1] * len(oos_records) + [0] * n_in
        oos_scores = [1 - x["record"]["top1_sim"] for x in oos_records] + \
                     [1 - x["record"]["top1_sim"] for x in in_dom]
        m2 = oos_metrics(oos_scores, oos_labels)
    else:
        oos_rej, m2 = None, None

    # === 输出 ===
    print(f"[{ds}] 统一口径 v1.0 (alpha={CFG['conformal']['alpha']}, full-level)")
    print(f"lambda={lam:.4f} empirical_risk={meta['empirical_risk']:.4f} theta_oos={theta_oos:.4f}")
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
    print(f"Decision dist:      {dict(Counter(x['decision'].state for x in results))}")


if __name__ == "__main__":
    main()
