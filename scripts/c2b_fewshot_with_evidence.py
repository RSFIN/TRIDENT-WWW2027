"""消融实验：给few-shot baseline补上和TRIDENT一样的检索evidence。

对比"无检索evidence的few-shot" vs "有检索evidence的few-shot"，
证明TRIDENT的优势不是因为用了额外的检索信息，而是共形校准方法本身。

用法：
    python scripts/c2b_fewshot_with_evidence.py --dataset clinc150 --split test --workers 10
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_fail_count = 0


class FewShotWithEvidenceClient(LLMClient):
    """Few-shot baseline，但是补上和TRIDENT一样的检索evidence。"""
    
    def classify_intent_fewshot(self, query: str, candidates: list, few_shot_examples: list, kb: dict) -> dict:
        """Few-shot prompt，加上检索到的候选intent evidence。"""
        # 先写few-shot示例
        few_shot_text = "\n\n".join(
            f"Example {i+1}:\nQuery: {ex['query']}\nIntent: {ex['intent']}"
            for i, ex in enumerate(few_shot_examples)
        )
        
        # 再加上检索到的候选intent的examples作为evidence（candidates是[(intent_name, score),...]）
        evidence_text = "\n".join(
            f"- {intent_name}: {kb[intent_name]['definition']}（示例：{'; '.join(kb[intent_name].get('examples', [])[:2])}）"
            for intent_name, score in candidates
        )
        
        system = (
            "You are an intent classifier. Given a user query and some example intents, "
            "identify the single most likely intent. Return ONLY the intent name."
        )
        user = f"""Few-shot examples:
{few_shot_text}

Additional candidate intents with definitions and examples:
{evidence_text}

Now classify this query:
Query: {query}

Return ONLY the intent name from the list above, nothing else."""
        
        out = self.chat(system, user)
        pred = out.strip().strip('"').strip("'")
        return {"main": pred, "aux": [], "confidence": {pred: 1.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["calib", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / args.dataset / "kb.json", encoding="utf-8"))
    idx = IntentIndex.load(str(ROOT / "data" / "indexes" / args.dataset), embedder, kb)
    llm = FewShotWithEvidenceClient(model=os.getenv("LLM_MODEL", CFG["llm"]["model"]))
    llm.temperature = 0.0  # few-shot baseline也是确定性的

    samples = load_jsonl(ROOT / "data" / "processed" / args.dataset / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]
    
    k_r = CFG["conformal"]["k_r_use"]
    print(f"Dataset: {args.dataset}, Split: {args.split}, Samples: {len(samples)}, K_r: {k_r}")

    # 准备few-shot示例：每个数据集取3个示例
    few_shot_examples = []
    kb_intents = list(kb.values())[:3]
    for intent in kb_intents:
        if intent.get("examples"):
            few_shot_examples.append({
                "query": intent["examples"][0],
                "intent": intent.get("intent", list(kb.keys())[list(kb.values()).index(intent)])
            })

    # 先做检索
    print("Step 1: Retrieving candidates...")
    candidates_list = []
    for s in samples:
        cands = idx.search(s.utterance, top_k=k_r)
        candidates_list.append(cands)

    # 并发跑LLM
    print(f"Step 2: Running few-shot LLM with evidence, {args.workers} workers...")
    results = [None] * len(samples)
    
    def process_one(idx):
        global _fail_count
        try:
            return idx, llm.classify_intent_fewshot(samples[idx].utterance, candidates_list[idx], few_shot_examples, kb)
        except Exception as e:
            _fail_count += 1
            print(f"  [FAIL] sample {idx}: {e}")
            return idx, {"main": "", "aux": [], "confidence": {}}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, i) for i in range(len(samples))]
        done = 0
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                speed = done / elapsed
                eta = (len(samples) - done) / speed / 60
                print(f"  Progress: {done}/{len(samples)} ({speed:.1f}/s), ETA {eta:.1f}min")

    elapsed = time.time() - t0
    print(f"Done! Total time: {elapsed:.1f}s, Failures: {_fail_count}")

    # 保存结果
    out_dir = ROOT / "data" / "predictions" / f"{args.dataset}_fewshot_ev"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}.jsonl"
    
    with open(out_file, "w", encoding="utf-8") as f:
        for s, r in zip(samples, results):
            f.write(json.dumps({
                "id": getattr(s, "id", ""),
                "text": s.utterance,
                "label": s.main_intent,
                "pred_main": r.get("main", ""),
                "pred_aux": r.get("aux", []),
                "confidence": r.get("confidence", {}),
            }, ensure_ascii=False) + "\n")
    
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
