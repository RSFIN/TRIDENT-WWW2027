"""调试：打印一条样本的 fused scores。"""
import sys, json, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.conformal.scores import f_ret, f_llm, fuse, rank_normalize

CFG = yaml.safe_load(open(ROOT/"configs/default.yaml", encoding="utf-8"))
data = [json.loads(l) for l in open(ROOT/"data/predictions/mixatis/calib.jsonl", encoding="utf-8")]

r = data[0]
print("utterance:", r["utterance"][:80])
print("true intents:", r["intents"], "main:", r["main_intent"])
print("llm_main:", r["llm_main"], "llm_aux:", r["llm_aux"])
print("retrieved:", r["retrieved"])

cands = [(it,s) for it,s in r["retrieved"]]
cand_ints = {it for it,_ in cands}
if r["llm_main"] not in cand_ints:
    cands.append((r["llm_main"], 0.0))
for a in (r["llm_aux"] or []):
    ex = {it for it,_ in cands}
    if a not in ex: cands.append((a, 0.0))

ret = f_ret(cands)
llm = f_llm(r, cands)
print("\nf_ret:", {k: round(v,3) for k,v in ret.items()})
print("f_llm:", {k: round(v,3) for k,v in llm.items()})

fa = CFG["conformal"]["fusion_alpha"]
fused = fuse(ret, llm, alpha=fa)
print(f"\nfused (alpha={fa}):")
for k,v in sorted(fused.items(), key=lambda x: x[1]):
    mark = " <-- pred_main" if k==sorted(fused.items(), key=lambda x:x[1])[0][0] else ""
    print(f"  {k}: {v:.3f}{mark}")
