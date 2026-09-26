"""快速扫描 fusion_alpha，找最优融合权重。"""
import sys, json, yaml
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold
from src.eval.metrics import macro_f1

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))

def load_pred(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

calib = load_pred(ROOT / "data/predictions/mixatis/calib.jsonl")
test  = load_pred(ROOT / "data/predictions/mixatis/test.jsonl")

for fa in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    calib_fused = []
    for r in calib:
        if r["oos"]:
            calib_fused.append({}); continue
        cands = [(it, s) for it, s in r["retrieved"]]
        ret = f_ret(cands); llm = f_llm(r, cands)
        calib_fused.append(fuse(ret, llm, alpha=fa))
    in_calib = [f for f,r in zip(calib_fused,calib) if not r["oos"]]
    main_c = [r["main_intent"] for r in calib if not r["oos"]]
    aux_c  = [[i for i in r["intents"] if i!=r["main_intent"]] for r in calib if not r["oos"]]
    lam, meta = calibrate_threshold(in_calib, main_c, aux_c,
        alpha=CFG["conformal"]["alpha"], tau_aux=1.0, k_max=3)

    # test
    test_main = []; pred_main = []
    for r in test:
        if r["oos"]: continue
        cands = [(it,s) for it,s in r["retrieved"]]
        ret = f_ret(cands); llm = f_llm(r, cands)
        fused = fuse(ret, llm, alpha=fa)
        m, a, sz = predict_set(fused, lam, 3)
        test_main.append(r["main_intent"])
        pred_main.append(m)
    f1 = macro_f1(test_main, pred_main)
    print(f"fusion_alpha={fa:.1f}  lambda={lam:.3f}  emp_risk={meta['empirical_risk']:.3f}  test_F1={f1:.3f}")
