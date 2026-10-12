import sys, json
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
import yaml
from src.conformal.scores import f_ret, f_llm, fuse
from src.conformal.rcps import calibrate_threshold, predict_set
from src.conformal.decision import decide, calibrate_oos_threshold

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
def load_pred(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
def build_fused_scores(records):
    all_fused = []
    for r in records:
        if r["oos"]:
            all_fused.append({}); continue
        candidates = [(it, sim) for it, sim in r["retrieved"]]
        cand_ints = {it for it, _ in candidates}
        if r.get("llm_main") and r["llm_main"] not in cand_ints:
            candidates.append((r["llm_main"], 0.0))
        for a in (r.get("llm_aux") or []):
            if a not in {it for it, _ in candidates}:
                candidates.append((a, 0.0))
        ret = f_ret(candidates); llm = f_llm(r, candidates)
        all_fused.append(fuse(ret, llm, alpha=CFG["conformal"]["fusion_alpha"]))
    return all_fused

test = load_pred(ROOT / "data/predictions/clinc150/test.jsonl")
calib = load_pred(ROOT / "data/predictions/clinc150/calib.jsonl")
cf = build_fused_scores(calib)
lam, meta = calibrate_threshold(
    [f for f,r in zip(cf,calib) if not r["oos"]],
    [r["main_intent"] for r in calib if not r["oos"]],
    [[i for i in r["intents"] if i!=r["main_intent"]] for r in calib if not r["oos"]],
    alpha=CFG["conformal"]["alpha"], tau_aux=CFG["conformal"]["tau_aux"], k_max=CFG["conformal"]["k_max"])
in_sims = [r["top1_sim"] for r in calib if not r["oos"]]
theta = calibrate_oos_threshold(in_sims, CFG["conformal"]["oos_fpr"])
print(f"lambda={lam:.4f} theta_oos={theta:.4f}")

tf = build_fused_scores(test)
reasons = {"oos_gate": 0, "set0": 0, "both": 0}
for r, fused in zip(test, tf):
    if r["oos"]: continue
    main, aux, sz = predict_set(fused, lam, CFG["conformal"]["k_max"])
    d = decide(main, aux, sz, r["top1_sim"], theta, CFG["conformal"]["k_max"])
    if d.state != "reject": continue
    gate = r["top1_sim"] < theta
    if gate and sz == 0: reasons["both"] += 1
    elif gate: reasons["oos_gate"] += 1
    else: reasons["set0"] += 1
print("in-domain reject 来源:", reasons)
