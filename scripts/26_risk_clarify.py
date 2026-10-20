"""Equal-risk fair comparison and oracle-clarification protocol.

Mode 1 (risk): 扫描 OOS gate 的 FPR 标定（oos_fpr），得到"不同 OOS 拒识率下的
F1/ECE/误拒率/IoU"权衡曲线，用于与少样本基线（OOS 拒识 83%-96%）公平对比。

Mode 2 (clarify): 把 clarify 决策视为"接受用户澄清后集合正确"（oracle），
对比现口径（clarify 集合 = 候选集）与 oracle 口径的集合指标，
回应"少样本基线直接输出大集合 vs Ours 澄清交互"的公平性问题。

用法（本地零 API）：
    python scripts/26_risk_clarify.py --mode risk --dataset clinc150 \
        --oos-fprs 0.05,0.02,0.01,0.005,0.002,0.001 --kr 1
    python scripts/26_risk_clarify.py --mode clarify --dataset mixatis --kr 1
    python scripts/26_risk_clarify.py --mode clarify --dataset mixsnips --kr 1
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


def build_fused_scores(records, fusion_alpha, kr):
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


def run_pipeline(ds, pred_subdir, fusion_alpha, kr, tau_aux, k_max, oos_fpr):
    """复用主配置流水线，返回 (results, lam, meta, theta_oos)。"""
    pred_dir = ROOT / "data" / "predictions" / ds if not pred_subdir \
        else ROOT / "data" / "predictions" / f"{ds}_{pred_subdir}"
    calib = load_pred(pred_dir / "calib.jsonl")
    test = load_pred(pred_dir / "test.jsonl")

    calib_fused = build_fused_scores(calib, fusion_alpha, kr)
    calib_main = [r["main_intent"] for r in calib if not r["oos"]]
    calib_aux = [[i for i in r["intents"] if i != r["main_intent"]]
                 for r in calib if not r["oos"]]
    calib_fused_in = [f for f, r in zip(calib_fused, calib) if not r["oos"]]

    lam, meta = calibrate_threshold(
        calib_fused_in, calib_main, calib_aux,
        alpha=CFG["conformal"]["alpha"], tau_aux=tau_aux, k_max=k_max,
    )

    in_dom_sims = [r["top1_sim"] for r in calib if not r["oos"]]
    theta_oos = calibrate_oos_threshold(in_dom_sims, oos_fpr)

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
    return results, lam, meta, theta_oos


def compute_metrics(results, oracle_clarify=False):
    """统一口径指标；oracle_clarify 时 clarify 样本集合替换为真实意图。"""
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
        else:  # clarify
            if oracle_clarify:
                pred_full.append(x["record"]["intents"])
                pred_ranked.append(x["record"]["intents"])
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
    oos_rej = None
    if oos_records:
        oos_rej = sum(1 for x in oos_records if x["decision"]["state"] == "reject") / len(oos_records)

    return {
        "f1": f1, "exact": exact, "iou": iou,
        "hit1": hit1, "hit3": hit3,
        "cov05": cov05, "cov_any": cov_any, "asize": asize,
        "ece": ece_val, "frr": frr, "clarify_rate": clarify_rate,
        "oos_rej": oos_rej,
        "dist": dict(Counter(x["decision"]["state"] for x in results)),
    }


def mode_risk(args):
    fprs = [float(x) for x in args.oos_fprs.split(",")]
    rows = []
    for fpr in fprs:
        results, lam, meta, theta = run_pipeline(
            args.dataset, args.pred_subdir,
            args.fusion_alpha, args.kr, args.tau_aux, args.k_max, fpr,
        )
        m = compute_metrics(results)
        rows.append((fpr, lam, theta, m))
    lines = [
        f"[risk_tradeoff_{args.dataset}] oos_fpr scan | kr={args.kr} k_max={args.k_max} "
        f"fa={args.fusion_alpha} tau={args.tau_aux} subdir={args.pred_subdir or 'default'}",
        f"{'oos_fpr':>8} {'lambda':>7} {'theta':>7} {'OOS拒识':>8} {'F1':>7} {'ECE':>7} "
        f"{'误拒率':>7} {'IoU':>7} {'Cov含':>7} {'avg|P|':>7}",
    ]
    for fpr, lam, theta, m in rows:
        oos_s = f"{m['oos_rej']:.4f}" if m["oos_rej"] is not None else "  N/A"
        lines.append(
            f"{fpr:>8.4f} {lam:>7.4f} {theta:>7.4f} {oos_s:>8} {m['f1']:>7.4f} "
            f"{m['ece']:>7.4f} {m['frr']:>7.4f} {m['iou']:>7.4f} "
            f"{m['cov_any']:>7.4f} {m['asize']:>7.2f}"
        )
        print(f"oos_fpr={fpr} theta={theta:.4f} OOS拒识={oos_s} F1={m['f1']:.4f} "
              f"ECE={m['ece']:.4f} 误拒={m['frr']:.4f} IoU={m['iou']:.4f}")
    out_path = ROOT / "results" / f"risk_tradeoff_{args.dataset}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {out_path}")


def mode_clarify(args):
    for oracle in (False, True):
        results, lam, meta, theta = run_pipeline(
            args.dataset, args.pred_subdir,
            args.fusion_alpha, args.kr, args.tau_aux, args.k_max,
            CFG["conformal"]["oos_fpr"],
        )
        m = compute_metrics(results, oracle_clarify=oracle)
        tag = "oracle" if oracle else "current"
        print(f"[{tag}] F1={m['f1']:.4f} Exact={m['exact']:.4f} IoU={m['iou']:.4f} "
              f"Cov(J>=0.5)={m['cov05']:.4f} Cov(含)={m['cov_any']:.4f} avg|P|={m['asize']:.2f} "
              f"澄清率={m['clarify_rate']:.4f} 误拒={m['frr']:.4f}")
    # 输出对比文件
    cur, ora = {}, {}
    for oracle in (False, True):
        results, lam, meta, theta = run_pipeline(
            args.dataset, args.pred_subdir,
            args.fusion_alpha, args.kr, args.tau_aux, args.k_max,
            CFG["conformal"]["oos_fpr"],
        )
        m = compute_metrics(results, oracle_clarify=oracle)
        (cur if not oracle else ora).update(m)
    out_path = ROOT / "results" / f"clarify_oracle_{args.dataset}.txt"
    lines = [
        f"[clarify_oracle_{args.dataset}] 现口径 vs oracle-clarify | kr={args.kr} "
        f"k_max={args.k_max} fa={args.fusion_alpha}",
        f"{'指标':<14} {'现口径':>10} {'oracle澄清':>10}",
        f"{'Main Macro-F1':<14} {cur['f1']:>10.4f} {ora['f1']:>10.4f}",
        f"{'Set Exact':<14} {cur['exact']:>10.4f} {ora['exact']:>10.4f}",
        f"{'Set IoU':<14} {cur['iou']:>10.4f} {ora['iou']:>10.4f}",
        f"{'Cov(J>=0.5)':<14} {cur['cov05']:>10.4f} {ora['cov05']:>10.4f}",
        f"{'Cov(含)':<14} {cur['cov_any']:>10.4f} {ora['cov_any']:>10.4f}",
        f"{'Avg set':<14} {cur['asize']:>10.2f} {ora['asize']:>10.2f}",
        f"{'澄清率':<14} {cur['clarify_rate']:>10.4f} {ora['clarify_rate']:>10.4f}",
        f"{'误拒率':<14} {cur['frr']:>10.4f} {ora['frr']:>10.4f}",
        f"{'ECE':<14} {cur['ece']:>10.4f} {ora['ece']:>10.4f}",
        "注：oracle 口径将 clarify 决策视为接受用户澄清后集合正确（集合=真实意图），"
        "用于评估澄清交互的系统价值上限。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["risk", "clarify"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-subdir", type=str, default="")
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--fusion-alpha", type=float, default=None)
    ap.add_argument("--kr", type=int, default=None)
    ap.add_argument("--tau-aux", type=float, default=None)
    ap.add_argument("--oos-fprs", type=str, default="0.05,0.02,0.01,0.005,0.002,0.001")
    args = ap.parse_args()

    args.k_max = args.k_max if args.k_max is not None else CFG["conformal"]["k_max"]
    args.fusion_alpha = args.fusion_alpha if args.fusion_alpha is not None \
        else CFG["conformal"]["fusion_alpha"]
    args.kr = args.kr if args.kr is not None else CFG["conformal"]["k_r_use"]
    args.tau_aux = args.tau_aux if args.tau_aux is not None else CFG["conformal"]["tau_aux"]

    if args.mode == "risk":
        mode_risk(args)
    else:
        mode_clarify(args)


if __name__ == "__main__":
    main()
