"""诊断：calib/test 上 LLM main 准确率、检索 top1 准确率、融合后 main 准确率。"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

for split in ["calib", "test"]:
    data = load(ROOT / f"data/predictions/mixatis/{split}.jsonl")
    in_d = [r for r in data if not r["oos"]]
    llm_correct = sum(1 for r in in_d if r["llm_main"] == r["main_intent"])
    ret_correct = sum(1 for r in in_d if r["retrieved"][0][0] == r["main_intent"])
    print(f"[{split}] n={len(in_d)}  LLM_main_acc={llm_correct/len(in_d):.3f}  ret_top1_acc={ret_correct/len(in_d):.3f}")
    # LLM aux 覆盖
    aux_hit = 0
    for r in in_d:
        true_aux = [i for i in r["intents"] if i != r["main_intent"]]
        if set(true_aux).issubset(set(r["llm_aux"])):
            aux_hit += 1
    print(f"       LLM_aux_recall_all={aux_hit/len(in_d):.3f}")
