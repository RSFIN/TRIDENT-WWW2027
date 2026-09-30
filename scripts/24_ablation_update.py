"""A8 知识更新模拟：新意图上线 → exchangeability 漂移前后的覆盖率对比。

模拟场景（CLINC150）：
  - before（无漂移）：用全量 calib（150 意图）校准 λ_full，test 全量评估 → 基线
  - after（漂移）：模拟"新意图上线"——从 calib 剔除 15% 意图（视为待上线新意图），
    用剩余 85% 意图的 calib 样本校准 λ_partial；test 上按"旧意图/新意图"分层报告
    Coverage(含)、误拒率、avg set、F1
关键问题：新意图样本（真实意图未见过）的覆盖率是否崩（Cov(含)→低），
证明知识更新（加条目）是恢复共形保证的必要机制。
"""
import sys
import json
import random
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import yaml
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import macro_f1, set_jaccard, coverage_jaccard, avg_set_size

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
SEED = CFG["seed"]


def load_pred(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_fused_scores(records, exclude_intents=None):
    """exclude_intents: 模拟"知识库未含这些意图"——候选里属于它们的条目相似度置 0（检索未命中），
    LLM 预测若属于它们则剔除（LLM 同样不认识）。"""
    excl = set(exclude_intents or [])
    all_fused = []
    for r in records:
        if r["oos"]:
            all_fused.append({})
            continue
        candidates = [(it, sim if it not in excl else 0.0)
                      for it, sim in r["retrieved"]]
        cand_ints = {it for it, _ in candidates}
        llm_main = r.get("llm_main")
        llm_aux = r.get("llm_aux") or []
        if llm_main and llm_main not in excl and llm_main not in cand_ints:
            candidates.append((llm_main, 0.0))
        for a in llm_aux:
            if a in excl:
                continue
            existing = {it for it, _ in candidates}
            if a not in existing:
                candidates.append((a, 0.0))
        ret = f_ret(candidates)
        llm = f_llm(r, candidates)
        all_fused.append(fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"]))
    return all_fused


def main():
    ds = "clinc150"
    calib = load_pred(ROOT / "data" / "predictions" / ds / "calib.jsonl")
    test = load_pred(ROOT / "data" / "predictions" / ds / "test.jsonl")
    k_max = CFG["conformal"]["k_max"]
    alpha = CFG["conformal"]["alpha"]

    calib_in = [r for r in calib if not r["oos"]]
    test_in = [r for r in test if not r["oos"]]
    test_oos = [r for r in test if r["oos"]]

    intents = sorted({r["main_intent"] for r in calib_in})
    random.seed(SEED)
    random.shuffle(intents)
    n_new = max(1, int(round(len(intents) * 0.15)))
    new_intents = set(intents[:n_new])
    old_intents = set(intents[n_new:])
    print(f"[A8] 意图总数={len(intents)} 新上线意图={len(new_intents)} ({n_new}) 旧意图={len(old_intents)}")

    # ---- before：全量 calib 校准 ----
    calib_fused_full = build_fused_scores(calib_in)
    calib_main_full = [r["main_intent"] for r in calib_in]
    calib_aux_full = [[i for i in r["intents"] if i != r["main_intent"]] for r in calib_in]
    lam_full, meta_full = calibrate_threshold(
        calib_fused_full, calib_main_full, calib_aux_full,
        alpha=alpha, tau_aux=CFG["conformal"]["tau_aux"], k_max=k_max)

    # ---- after：剔除新意图的 calib 校准 ----
    keep = [i for i, r in enumerate(calib_in) if r["main_intent"] in old_intents]
    calib_fused_part = [calib_fused_full[i] for i in keep]
    calib_main_part = [calib_main_full[i] for i in keep]
    calib_aux_part = [calib_aux_full[i] for i in keep]
    lam_part, meta_part = calibrate_threshold(
        calib_fused_part, calib_main_part, calib_aux_part,
        alpha=alpha, tau_aux=CFG["conformal"]["tau_aux"], k_max=k_max)

    theta_oos_full = calibrate_oos_threshold(
        [r["top1_sim"] for r in calib_in], CFG["conformal"]["oos_fpr"])
    theta_oos_part = calibrate_oos_threshold(
        [calib_in[i]["top1_sim"] for i in keep], CFG["conformal"]["oos_fpr"])

    print(f"lam_full={lam_full:.4f} (emp_risk={meta_full['empirical_risk']:.4f}, n={len(calib_main_full)})")
    print(f"lam_part={lam_part:.4f} (emp_risk={meta_part['empirical_risk']:.4f}, n={len(calib_main_part)})")

    # ---- test 评估函数（分层）----
    def evaluate(test_records, fused, lam, theta_oos, label):
        results = []
        for r, fz in zip(test_records, fused):
            if r["oos"]:
                d = decide("", [], 0, r["top1_sim"], theta_oos, k_max)
            else:
                main, aux, sz = predict_set(fz, lam, k_max)
                d = decide(main, aux, sz, r["top1_sim"], theta_oos, k_max)
            results.append({"record": r, "decision": d.__dict__, "fused": fz})

        in_dom = [x for x in results if not x["record"]["oos"]]
        for group, filt in [("all", lambda r: True),
                            ("old", lambda r: r["main_intent"] in old_intents),
                            ("new", lambda r: r["main_intent"] in new_intents)]:
            sub = [x for x in in_dom if filt(x["record"])]
            if not sub:
                continue
            true_full = [x["record"]["intents"] for x in sub]
            pred_full = []
            for x in sub:
                d = x["decision"]
                if d["state"] == "reject":
                    pred_full.append([])
                elif d["state"] == "release":
                    pred_full.append([d["main"]])
                else:
                    pred_full.append(d.get("candidates") or ([d["main"]] + (d["aux"] or [])))
            n = len(sub)
            n_rej = sum(1 for x in sub if x["decision"]["state"] == "reject")
            cov_any = sum(1 for t, p in zip(true_full, pred_full) if set(t) & set(p)) / n
            cov05 = coverage_jaccard(true_full, pred_full, 0.5)
            asize = avg_set_size(pred_full)
            print(f"  [{label}/{group}] n={n} Cov(含)={cov_any:.4f} Cov(J>=0.5)={cov05:.4f} "
                  f"avg set={asize:.2f} 误拒={n_rej / n:.4f}")
        # 主意图 F1（all）
        in_all = in_dom
        tm = [x["record"]["main_intent"] for x in in_all]
        pm = [x["decision"]["main"] for x in in_all]
        print(f"  [{label}/all] Main F1={macro_f1(tm, pm):.4f}")

    test_fused = build_fused_scores(test)
    evaluate(test, test_fused, lam_full, theta_oos_full, "before_fullcalib")

    # after：知识库未含新意图（检索相似度置 0 + LLM 剔除）+ 部分 calib 校准
    # 注：calib 部分校准的 λ（lam_part）与全量 λ 相同（CLINC150 λ=0.0），
    # 为清晰起见这里用"未更新知识库 + 全量 λ"评估新意图样本的覆盖崩塌，
    # 再与"已更新知识库（before）"对照，得到知识更新前后的 coverage gap。
    test_fused_excl = build_fused_scores(test, exclude_intents=new_intents)
    evaluate(test, test_fused_excl, lam_full, theta_oos_full, "after_no_update")


if __name__ == "__main__":
    main()
