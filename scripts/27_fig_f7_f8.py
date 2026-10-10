"""F7: alpha sweep coverage vs set size (k_max=3 to allow set growth).
F8: OOS ROC (retrieval vs LLM conf). Pure local, zero API.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml, numpy as np
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.eval.metrics import coverage_jaccard, avg_set_size

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
TAU = CFG["conformal"]["tau_aux"]
FA = CFG["conformal"]["fusion_alpha"]

def load_pred(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def build_fused(records):
    out = []
    for r in records:
        if r["oos"]:
            out.append({}); continue
        cands = [(it, s) for it, s in r["retrieved"]]
        ex = {i for i,_ in cands}
        if r.get("llm_main") and r["llm_main"] not in ex:
            cands.append((r["llm_main"], 0.0))
        for a in (r.get("llm_aux") or []):
            if a not in {i for i,_ in cands}:
                cands.append((a, 0.0))
        out.append(fuse(f_ret(cands), f_llm(r, cands), alpha=FA))
    return out

alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
K_MAX = 3  # allow set growth to show trade-off
f7 = {}
for ds in ["clinc150", "mixatis", "mixsnips"]:
    calib = load_pred(ROOT/"data"/"predictions"/ds/"calib.jsonl")
    test = load_pred(ROOT/"data"/"predictions"/ds/"test.jsonl")
    calib_f = build_fused(calib)
    calib_in = [(f, r) for f, r in zip(calib_f, calib) if not r["oos"]]
    calib_main = [r["main_intent"] for _, r in calib_in]
    calib_aux = [[i for i in r["intents"] if i != r["main_intent"]] for _, r in calib_in]
    calib_f_in = [f for f, _ in calib_in]
    test_f = build_fused(test)
    is_single = (ds == "clinc150")  # single-intent dataset

    rows = []
    for a in alphas:
        lam, meta = calibrate_threshold(calib_f_in, calib_main, calib_aux,
                                        alpha=a, tau_aux=TAU, k_max=K_MAX)
        true_aux_all, pred_full_all, main_hit = [], [], 0
        n_in = 0
        for r, fused in zip(test, test_f):
            if r["oos"]: continue
            n_in += 1
            main, aux, sz = predict_set(fused, lam, K_MAX)
            pf = [main] + (aux or []) if main else (aux or [])
            if is_single:
                if main == r["main_intent"]:
                    main_hit += 1
            else:
                true_aux_all.append([i for i in r["intents"] if i != r["main_intent"]])
                pred_full_all.append(pf)
        if is_single:
            cov = main_hit / max(n_in,1)
        else:
            cov = coverage_jaccard(true_aux_all, pred_full_all, 0.5)
        # avg set size over in-domain
        sizes = []
        for r, fused in zip(test, test_f):
            if r["oos"]: continue
            main, aux, sz = predict_set(fused, lam, K_MAX)
            pf = [main] + (aux or []) if main else (aux or [])
            sizes.append(len(pf))
        sz = float(np.mean(sizes))
        rows.append({"alpha": a, "nominal_cov": 1-a, "emp_cov": cov, "avg_set": sz})
        print(f"[F7] {ds} alpha={a:.2f} emp_cov={cov:.4f} avg_set={sz:.2f}")
    f7[ds] = rows

json.dump(f7, open(ROOT/"results"/"_f7_alpha_sweep.json","w",encoding="utf-8"), indent=2)

# F8 OOS ROC
test = load_pred(ROOT/"data"/"predictions"/"clinc150"/"test.jsonl")
ret_score, llm_score, labels = [], [], []
for r in test:
    labels.append(1 if r["oos"] else 0)
    ret_score.append(1.0 - r["top1_sim"])
    confs = r.get("llm_confidence") or {}
    mx = max(confs.values()) if confs else 0.0
    llm_score.append(1.0 - mx)

from sklearn.metrics import roc_curve, auc
def roc(score):
    fpr, tpr, _ = roc_curve(labels, score)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))

rf, rt, ra = roc(ret_score)
lf, lt, la = roc(llm_score)
f8 = {"retrieval": {"fpr": rf, "tpr": rt, "auroc": ra},
      "llm_conf":  {"fpr": lf, "tpr": lt, "auroc": la}}
print(f"[F8] retrieval AUROC={ra:.4f}  LLM-conf AUROC={la:.4f}")
json.dump(f8, open(ROOT/"results"/"_f8_roc.json","w",encoding="utf-8"))
print("DONE")
