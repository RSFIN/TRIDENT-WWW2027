"""对 train/calib/test 跑 LLM 推理，缓存结果到 data/predictions/。

用法：
    python scripts/03_run_llm.py --dataset clinc150 --split test
"""
import sys
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import yaml
from src.data.schema import IntentSample
from src.data.prepare import load_jsonl
from src.retriever.embedder import Embedder
from src.retriever.index import IntentIndex
from src.llm.client import LLMClient

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "calib", "test"])
    ap.add_argument("--limit", type=int, default=None, help="调试时限制样本数")
    args = ap.parse_args()

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / args.dataset / "kb.json", encoding="utf-8"))
    idx = IntentIndex.load(str(ROOT / "data" / "indexes" / args.dataset), embedder, kb)
    llm = LLMClient(model=CFG["llm"]["model"])

    samples = load_jsonl(ROOT / "data" / "processed" / args.dataset / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]

    out_path = ROOT / "data" / "predictions" / args.dataset / f"{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, s in enumerate(samples):
            if s.oos:
                # OOS 不跑 LLM，只记 top1_sim
                top1 = idx.top1_sim(s.utterance)
                rec = {
                    "utterance": s.utterance, "intents": s.intents,
                    "main_intent": s.main_intent, "oos": True,
                    "top1_sim": top1,
                    "llm_main": "", "llm_aux": [], "llm_confidence": {},
                    "retrieved": [],
                }
            else:
                cands = idx.search(s.utterance, top_k=CFG["retriever"]["top_k_r"])
                cand_dicts = [
                    {"intent": it, "definition": kb[it]["definition"],
                     "examples": kb[it]["examples"]}
                    for it, _ in cands
                ]
                result = llm.classify_intent(s.utterance, cand_dicts)
                top1 = cands[0][1] if cands else 0.0
                rec = {
                    "utterance": s.utterance, "intents": s.intents,
                    "main_intent": s.main_intent, "oos": False,
                    "top1_sim": top1,
                    "retrieved": [(it, sim) for it, sim in cands],
                    "llm_main": result.get("main", ""),
                    "llm_aux": result.get("aux", []),
                    "llm_confidence": result.get("confidence", {}),
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  [{args.dataset}/{args.split}] {i+1}/{len(samples)}")
                fout.flush()
            time.sleep(0.05)  # 限流

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
