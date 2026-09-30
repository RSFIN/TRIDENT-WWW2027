"""A9 跨域泛化：CLINC150 校准 → MixSNIPS 测试。

对比两种校准源在 MixSNIPS test 上的表现：
  - cross: λ 来自 CLINC150 calib（跨域，验证共形保证对同域分布的依赖）
  - same:  λ 来自 MixSNIPS calib（同域，作为对照）

评估统一口径全套指标（MixSNIPS 无 OOS，OOS gate 不参与）。
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
    coverage_jaccard, avg_set_size, ece,
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


def calibrate_lambda(calib_fused, calib_main, calib_aux):
    lam, meta = calibrate_threshold(
        calib_fused, calib_main, calib_aux,
        alpha=CFG["conformal"]["alpha"],
        tau_aux=CFG["conformal"]["tau_aux"],
        k_max=CFG["conformal"]["k_max"],
    )
    return lam, meta


def evaluate(test, test_fused, lam, theta_oos, tag):
    k_max = CFG["conformal"]["k_max"]
    results = []
    for r, fused in zip(test, test_fused):
        main, aux, sz = predict_set(fused, lam, k_max)
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, k_max)
        results.append({"record": r, "decision": d.__dict__, "fused": fused})

    in_dom = results
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

    lines = [
        f"[A9 {tag}] lambda={lam:.4f} alpha={CFG['conformal']['alpha']}",
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
        f"Decision dist:      {dict(Counter(x['decision']['state'] for x in in_dom))}",
    ]
    print("\n".join(lines))
    with open(ROOT / "results" / f"ablation_crossdomain_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ds_src, ds_tgt = "clinc150", "mixsnips"

    clinc_calib = load_pred(ROOT / "data" / "predictions" / ds_src / "calib.jsonl")
    mixs_calib = load_pred(ROOT / "data" / "predictions" / ds_tgt / "calib.jsonl")
    mixs_test = load_pred(ROOT / "data" / "predictions" / ds_tgt / "test.jsonl")

    # 跨域 λ：CLINC150 calib（in-domain 样本）
    clinc_fused = build_fused_scores(clinc_calib)
    clinc_in = [f for f, r in zip(clinc_fused, clinc_calib) if not r["oos"]]
    clinc_main = [r["main_intent"] for r in clinc_calib if not r["oos"]]
    clinc_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                 for r in clinc_calib if not r["oos"]]
    lam_cross, meta_cross = calibrate_lambda(clinc_in, clinc_main, clinc_aux)

    # 同域 λ：MixSNIPS calib
    mixs_fused_calib = build_fused_scores(mixs_calib)
    mixs_main = [r["main_intent"] for r in mixs_calib]
    mixs_aux = [[i for i in r["intents"] if i != r["main_intent"]] for r in mixs_calib]
    lam_same, meta_same = calibrate_lambda(mixs_fused_calib, mixs_main, mixs_aux)

    print(f"lambda_cross(CLINC150)={lam_cross:.4f} emp_risk={meta_cross['empirical_risk']:.4f}")
    print(f"lambda_same(MixSNIPS) ={lam_same:.4f} emp_risk={meta_same['empirical_risk']:.4f}")

    mixs_test_fused = build_fused_scores(mixs_test)
    # OOS gate 与校准同域：cross 用 CLINC150 calib 的 θ_oos，same 用 MixSNIPS calib 的 θ_oos
    clinc_theta = calibrate_oos_threshold(
        [r["top1_sim"] for r in clinc_calib if not r["oos"]],
        CFG["conformal"]["oos_fpr"])
    mixs_theta = calibrate_oos_threshold(
        [r["top1_sim"] for r in mixs_calib],
        CFG["conformal"]["oos_fpr"])
    print(f"theta_oos_cross(CLINC150)={clinc_theta:.4f}  theta_oos_same(MixSNIPS)={mixs_theta:.4f}")
    evaluate(mixs_test, mixs_test_fused, lam_cross, clinc_theta, "cross_clinc150_to_mixsnips")
    evaluate(mixs_test, mixs_test_fused, lam_same, mixs_theta, "same_mixsnips")


if __name__ == "__main__":
    main()
