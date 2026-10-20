"""Ablation under the unified protocol over K (k_max), fusion_alpha, and K_r (retrieval truncation).

用法：
    python scripts/20_ablation_unified.py --dataset clinc150 --k-max 0
    python scripts/20_ablation_unified.py --dataset mixsnips --fusion-alpha 1.0
    python scripts/20_ablation_unified.py --dataset mixatis --kr 1

输出统一口径 v1.0 全套指标（与 05_eval_unified.py 一致），结果写入 results/ablation_*。
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


def build_fused_scores(records, fusion_alpha, kr):
    """kr: 检索候选截断数；LLM 预测的 main/aux 强制加入候选集。"""
    all_fused = []
    for r in records:
        if r["oos"]:
            all_fused.append({})
            continue
        candidates = [(it, sim) for it, sim in r["retrieved"][:kr]] if kr else \
                     [(it, sim) for it, sim in r["retrieved"]]
        cand_ints = {it for it, _ in candidates}
        if r.get("llm_main") and r["llm_main"] not in cand_ints:
            candidates.append((r["llm_main"], 0.0))
        for a in (r.get("llm_aux") or []):
            existing = {it for it, _ in candidates}
            if a not in existing:
                candidates.append((a, 0.0))
        ret = f_ret(candidates)
        llm = f_llm(r, candidates)
        all_fused.append(fuse(ret, llm, alpha=fusion_alpha))
    return all_fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--fusion-alpha", type=float, default=None)
    ap.add_argument("--kr", type=int, default=None)
    ap.add_argument("--tau-aux", type=float, default=None)
    ap.add_argument("--pred-subdir", type=str, default="")
    args = ap.parse_args()
    ds = args.dataset

    k_max = args.k_max if args.k_max is not None else CFG["conformal"]["k_max"]
    fusion_alpha = args.fusion_alpha if args.fusion_alpha is not None \
        else CFG["conformal"]["fusion_alpha"]
    kr = args.kr
    tau_aux = args.tau_aux if args.tau_aux is not None else CFG["conformal"]["tau_aux"]

    pred_dir = ROOT / "data" / "predictions" / ds if not args.pred_subdir \
        else ROOT / "data" / "predictions" / f"{ds}_{args.pred_subdir}"
    calib = load_pred(pred_dir / "calib.jsonl")
    test = load_pred(pred_dir / "test.jsonl")

    # === 校准 ===
    calib_fused = build_fused_scores(calib, fusion_alpha, kr)
    calib_main = [r["main_intent"] for r in calib if not r["oos"]]
    calib_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                 for r in calib if not r["oos"]]
    calib_fused_in = [f for f, r in zip(calib_fused, calib) if not r["oos"]]

    lam, meta = calibrate_threshold(
        calib_fused_in, calib_main, calib_aux,
        alpha=CFG["conformal"]["alpha"],
        tau_aux=tau_aux,
        k_max=k_max,
    )

    in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
    theta_oos = calibrate_oos_threshold(in_dom_sims, CFG["conformal"]["oos_fpr"])

    # === 测试推理 ===
    test_fused = build_fused_scores(test, fusion_alpha, kr)
    results = []
    for r, fused in zip(test, test_fused):
        if r["oos"]:
            d = decide("", [], 0, r["top1_sim"], theta_oos, k_max)
            results.append({"record": r, "decision": d.__dict__, "fused": fused})
            continue
        main, aux, sz = predict_set(fused, lam, k_max)
        d = decide(main, aux, sz, r["top1_sim"], theta_oos, k_max)
        results.append({"record": r, "decision": d.__dict__, "fused": fused})

    # === 统一口径指标 ===
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
    frr = n_reject_in / n_in
    clarify_rate = n_clarify / n_in

    oos_records = [x for x in results if x["record"]["oos"]]
    if oos_records:
        oos_rej = sum(1 for x in oos_records if x["decision"]["state"] == "reject") / len(oos_records)
        oos_labels = [1] * len(oos_records) + [0] * n_in
        oos_scores = [1 - x["record"]["top1_sim"] for x in oos_records] + \
                     [1 - x["record"]["top1_sim"] for x in in_dom]
        m2 = oos_metrics(oos_scores, oos_labels)
    else:
        oos_rej, m2 = None, None

    tag = f"ablation_{ds}_k{k_max}_fa{fusion_alpha}"
    tag += f"_tau{tau_aux}" if args.tau_aux is not None else ""
    tag += f"_kr{kr}" if kr else ""
    tag += f"_{args.pred_subdir}" if args.pred_subdir else ""
    out_lines = [
        f"[{tag}] alpha={CFG['conformal']['alpha']}, full-level 统一口径",
        f"k_max={k_max} fusion_alpha={fusion_alpha:.3f} tau_aux={tau_aux} kr={kr if kr else 'full'} pred_subdir={args.pred_subdir or 'default'}",
        f"lambda={lam:.4f} emp_risk={meta['empirical_risk']:.4f} theta_oos={theta_oos:.4f}",
        f"Main Macro-F1:      {f1:.4f}",
        f"Set Exact Match:    {exact:.4f}",
        f"Set IoU:            {iou:.4f}",
        f"Hit@1/3/5:          {hit1:.4f} / {hit3:.4f} / {hit5:.4f}",
        f"Coverage(J>=0.5):   {cov05:.4f}",
        f"Coverage(含):       {cov_any:.4f}",
        f"Avg set size:       {asize:.2f}",
        f"ECE (15 bins):      {ece_val:.4f}",
        f"误拒率:             {frr:.4f} ({n_reject_in}/{n_in})",
        f"澄清触发率:         {clarify_rate:.4f}",
    ]
    if oos_records:
        out_lines += [
            f"OOS 拒识率:         {oos_rej:.4f}",
            f"OOS AUROC/AUPRC:    {m2['auroc']:.4f} / {m2['auprc']:.4f}",
        ]
    out_lines.append(f"Decision dist:      {dict(Counter(x['decision']['state'] for x in results))}")

    text = "\n".join(out_lines)
    print(text)
    with open(ROOT / "results" / f"{tag}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
