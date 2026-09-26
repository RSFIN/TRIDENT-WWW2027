"""调试校准集 loss 分布。"""
import sys, json, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import stratified_loss

CFG = yaml.safe_load(open(ROOT/"configs/default.yaml", encoding="utf-8"))
calib = [json.loads(l) for l in open(ROOT/"data/predictions/mixatis/calib.jsonl", encoding="utf-8")]

losses = []
main_correct = 0
main_in_set = 0
for r in calib:
    if r["oos"]: continue
    cands = [(it,s) for it,s in r["retrieved"]]
    ci = {it for it,_ in cands}
    if r["llm_main"] not in ci: cands.append((r["llm_main"], 0.0))
    for a in (r.get("llm_aux") or []):
        ex = {it for it,_ in cands}
        if a not in ex: cands.append((a, 0.0))
    ret = f_ret(cands); llm = f_llm(r, cands)
    fused = fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"])
    # lambda=1.0: 所有候选都收入
    pred = {i:s for i,s in fused.items() if s <= 1.0}
    sorted_i = sorted(pred.items(), key=lambda x: x[1])
    pm = sorted_i[0][0]
    pa = [i for i,_ in sorted_i[1:4]]
    ta = [i for i in r["intents"] if i != r["main_intent"]]
    l = stratified_loss(pm, pa, r["main_intent"], ta, tau_aux=0.5)
    losses.append(l)
    if pm == r["main_intent"]: main_correct += 1
    if r["main_intent"] in {pm}|set(pa): main_in_set += 1

import numpy as np
from collections import Counter
print(f"n={len(losses)}")
print(f"main_correct={main_correct/len(losses):.3f}")
print(f"main_in_set={main_in_set/len(losses):.3f}")
print(f"loss dist: {Counter(losses)}")
print(f"mean loss={np.mean(losses):.3f}")
