"""TECP-style token-entropy conformal prediction.

对每条 query 用 temperature>0 采样 LLM T 次，收集 main intent 分布，
后续在 11_eval_tecp.py 里算 entropy 做 split CP。

用法：
    python scripts/11_run_llm_samples.py --dataset clinc150 --split test --num-samples 5 --limit 500
"""
import sys, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import yaml
from src.data.prepare import load_jsonl
from src.retriever.embedder import Embedder
from src.retriever.index import IntentIndex
from src.llm.client import LLMClient

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["calib", "test"])
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / args.dataset / "kb.json", encoding="utf-8"))
    idx = IntentIndex.load(str(ROOT / "data" / "indexes" / args.dataset), embedder, kb)
    # 每次采样都用独立 client，temperature 设高
    llm = LLMClient(model=CFG["llm"]["model"])
    llm.temperature = args.temperature

    samples = load_jsonl(ROOT / "data" / "processed" / args.dataset / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]

    out_path = ROOT / "data" / "predictions" / args.dataset / f"tecp_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path, "r", encoding="utf-8"):
            try: done.add(json.loads(line)["utterance"])
            except: pass
    if done: print(f"[resume] {len(done)} done")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, s in enumerate(samples):
            if s.utterance in done: continue
            if s.oos:
                rec = {"utterance": s.utterance, "intents": s.intents,
                       "main_intent": s.main_intent, "oos": True, "samples": []}
            else:
                cands = idx.search(s.utterance, top_k=CFG["retriever"]["top_k_r"])
                cand_dicts = [{"intent": it, "definition": kb[it]["definition"],
                               "examples": kb[it]["examples"]} for it, _ in cands]
                mains = []
                for _ in range(args.num_samples):
                    r = llm.classify_intent(s.utterance, cand_dicts)
                    mains.append(r.get("main", ""))
                    time.sleep(0.05)
                rec = {"utterance": s.utterance, "intents": s.intents,
                       "main_intent": s.main_intent, "oos": False,
                       "samples": mains, "retrieved": [(it, sim) for it, sim in cands]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i+1) % 20 == 0:
                print(f"  [{args.dataset}/{args.split}] {i+1}/{len(samples)}")
                fout.flush()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
