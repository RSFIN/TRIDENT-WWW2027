"""A4 集合级风险度量消融：单点覆盖 vs Jaccard 覆盖 vs 对称差惩罚。

三种风险度量定义（对每样本预测集 P 与真实意图集 Y）：
  - single_point: L = 1[true_main ∉ P]                    （仅保证主意图，{0,1}）
  - jaccard:      L = 1 - Jaccard(P, Y)                    （集合级连续）
  - sym_diff:     L = |P△Y| / |P∪Y|                        （对称差惩罚，集合级连续）
reject（P=∅）时：single_point L=1（main 未覆盖）；jaccard L=1；sym_diff L=1。

对每种度量在校准集上选最小 λ 使经验风险 ≤ α，再统一口径评估 test。
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
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import jaccard, predict_set
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
        all_fused.append(fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"]))
    return all_fused


def risk_loss(loss_mode, pred_main, pred_aux, true_main, true_aux):
    pred_set = {pred_main} | set(pred_aux)
    true_set = {true_main} | set(true_aux)
    if loss_mode == "single_point":
        return 0.0 if true_main in pred_set else 1.0
    if loss_mode == "jaccard":
        return 1.0 - jaccard(list(pred_set), list(true_set))
    if loss_mode == "sym_diff":
        union = pred_set | true_set
        if not union:
            return 0.0
        inter = pred_set & true_set
        return (len(pred_set) + len(true_set) - 2 * len(inter)) / len(union)
    raise ValueError(loss_mode)


def calibrate_risk(calib_fused, calib_main, calib_aux, loss_mode, alpha, tau_aux, k_max):
    all_scores = sorted({s for d in calib_fused for s in d.values() if s < float("inf")})
    candidates = [0.0] + all_scores + [1.0]
    best_lam, best_risk = 1.0, 1.0
    for lam in candidates:
        losses = []
        for d, true_m, true_a in zip(calib_fused, calib_main, calib_aux):
            pred = {i: s for i, s in d.items() if s <= lam and s < float("inf")}
            if not pred:
                losses.append(1.0)
                continue
            sorted_ints = sorted(pred.items(), key=lambda x: x[1])
            pm = sorted_ints[0][0]
            pa = [i for i, _ in sorted_ints[1:k_max + 1]]
            losses.append(risk_loss(loss_mode, pm, pa, true_m, true_a))
        risk = np.mean(losses)
        if risk <= alpha:
            best_lam, best_risk = lam, risk
            break
    return best_lam, {"empirical_risk": best_risk, "n_calib": len(calib_fused)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", required=True, choices=["single_point", "jaccard", "sym_diff"])
    args = ap.parse_args()
    ds, loss_mode = args.dataset, args.loss

    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")

    k_max = CFG["conformal"]["k_max"]
    alpha = CFG["conformal"]["alpha"]
    tau_aux = CFG["conformal"]["tau_aux"]

    calib_fused = build_fused_scores(calib)
    calib_main = [r["main_intent"] for r in calib if not r["oos"]]
    calib_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                 for r in calib if not r["oos"]]
    calib_fused_in = [f for f, r in zip(calib_fused, calib) if not r["oos"]]

    lam, meta = calibrate_risk(calib_fused_in, calib_main, calib_aux, loss_mode, alpha, tau_aux, k_max)

    in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
    theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

    test_fused = build_fused_scores(test)
    results = []
    for r, fused in zip(test, test_fused):
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, k_max)
            results.append({"record": r, "decision": d.__dict__, "fused": fused})
            continue
        main, aux, sz = predict_set(fused, lam, k_max)
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, k_max)
        results.append({"record": r, "decision": d.__dict__, "fused": fused})

    in_dom = [x for x in results if not x["record"]["oos"]]
    true_full = [x["record"]["intents"] for x in in_dom]
    pred_full, pred_ranked = [], []
    for x in in_dom:
        d = x["decision"]
        if d["state"] == "reject":
            pred_full.append([])
            pred_ranked.append([])
        elif d["state"] == "release":
            pred_full.append([d["main"]])
            pred_ranked.append([d["main"]])
        else:
            cands = d.get("candidates") or ([d["main"]] + (d["aux"] or []))
            pred_full.append(cands)
            pred_ranked.append(cands)

    true_main = [x["record"]["main_intent"] for x in in_dom]
    pred_main = [x["decision"]["main"] for x in in_dom]

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
        if d["state"] == "reject":
            confs.append(0.0)
            corrects.append(0)
        else:
            fused = x["fused"]
            conf = 1.0 - min(fused.values()) if fused else 0.0
            confs.append(conf)
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

    tag = f"ablation_risk_{ds}_{loss_mode}"
    lines = [
        f"[A4 {tag}] loss={loss_mode} alpha={alpha}",
        f"lambda={lam:.4f} emp_risk={meta['empirical_risk']:.4f} theta_oos={theta_oos:.4f}",
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
