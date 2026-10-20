"""消融实验：去掉训练utterance，只用intent definition做检索和prompt。

用法：
    python scripts/c2a_no_examples_ablation.py --dataset clinc150 --split test --workers 10
"""
import sys
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
from src.data.schema import IntentSample
from src.data.prepare import load_jsonl
from src.retriever.embedder import Embedder
from src.llm.client import LLMClient, LLMClient as BaseLLMClient

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))

_fail_count = 0


class NoExamplesLLMClient(BaseLLMClient):
    """去掉examples的LLM client，prompt里只给intent name和definition。"""
    
    def classify_intent(self, query: str, candidates: list) -> dict:
        cand_text = "\n".join(
            f"- {c['intent']}: {c['definition']}"
            for c in candidates
        )
        system = (
            "You are an intent classifier. Given a user query and candidate intents, "
            "identify the MAIN intent and any AUXILIARY intents that also apply. "
            "Return STRICT JSON only."
        )
        user = f"""Query: {query}

Candidate intents:
{cand_text}

Return JSON:
{{"main": "<one intent>", "aux": ["<other intent that also applies, can be empty>"], "confidence": {{"<intent>": <0.0-1.0>, ...}}}}"""
        out = self.chat(system, user)
        return self._parse_json(out)


def build_definition_index(kb: dict, embedder: Embedder):
    """只用每个intent的definition建索引，不用训练utterance。"""
    # kb是dict: {intent_name: {"definition": str, "examples": list}}
    kb_def_only = []
    for intent_name, intent_info in kb.items():
        kb_def_only.append({
            "intent": intent_name,
            "definition": intent_info["definition"],
            "examples": [],  # 空，不用examples
        })
    
    # 用definition建索引
    texts = [f"{item['intent']}: {item['definition']}" for item in kb_def_only]
    embeddings = embedder.encode(texts)
    
    return kb_def_only, embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["calib", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / args.dataset / "kb.json", encoding="utf-8"))
    
    # 用definition建索引（不用训练utterance）
    kb_def, def_embeddings = build_definition_index(kb, embedder)
    
    llm = NoExamplesLLMClient()  # 用环境变量里的LLM_MODEL
    llm.temperature = CFG["llm"]["temperature"]
    llm.max_tokens = CFG["llm"]["max_tokens"]

    samples = load_jsonl(ROOT / "data" / "processed" / args.dataset / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]
    
    k_r = CFG["conformal"]["k_r_use"]
    print(f"Dataset: {args.dataset}, Split: {args.split}, Samples: {len(samples)}, K_r: {k_r}")

    # 先做检索（串行，避免多线程编码死锁）
    print("Step 1: Retrieving candidates using definition-only index...")
    queries = [s.utterance for s in samples]
    query_embs = embedder.encode(queries)
    
    import numpy as np
    candidates_list = []
    for i in range(len(samples)):
        # 余弦相似度
        sims = query_embs[i] @ def_embeddings.T
        top_idx = np.argsort(-sims)[:k_r]
        cands = [kb_def[j] for j in top_idx]
        candidates_list.append(cands)

    # 输出文件路径，支持断点续跑
    out_dir = ROOT / "data" / "predictions" / f"{args.dataset}_noex"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}.jsonl"

    # 读取已经完成的样本，跳过
    done_utterances = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done_utterances.add(rec["utterance"])
                    except:
                        pass
        print(f"Found existing results: {len(done_utterances)} samples already done, will skip them.")

    # 构造待跑任务：只跑还没完成的
    todo_indices = [i for i in range(len(samples)) if samples[i].utterance not in done_utterances]
    total_todo = len(todo_indices)
    print(f"Step 2: Running LLM inference with {args.workers} workers, {total_todo}/{len(samples)} samples to run...")

    # 打开文件追加写，每完成一个样本立刻落盘
    file_mode = "a" if out_file.exists() else "w"
    out_f = open(out_file, file_mode, encoding="utf-8")

    def process_one(idx):
        global _fail_count
        try:
            return idx, llm.classify_intent(samples[idx].utterance, candidates_list[idx])
        except Exception as e:
            _fail_count += 1
            print(f"  [FAIL] sample {idx}: {e}")
            return idx, {"main": "", "aux": [], "confidence": {}}

    t0 = time.time()
    done = 0
    if total_todo > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, i) for i in todo_indices]
            for fut in as_completed(futures):
                idx, res = fut.result()
                s = samples[idx]
                # 立刻写入文件并flush，防止崩溃丢数据
                out_f.write(json.dumps({
                    "utterance": s.utterance,
                    "main_intent": s.main_intent,
                    "intents": s.intents,
                    "oos": s.oos,
                    "pred_main": res.get("main", ""),
                    "pred_aux": res.get("aux", []),
                    "confidence": res.get("confidence", {}),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    speed = done / elapsed
                    eta = (total_todo - done) / speed / 60
                    print(f"  Progress: {done}/{total_todo} ({speed:.1f}/s), ETA {eta:.1f}min")

    out_f.close()
    elapsed = time.time() - t0
    print(f"Done! Newly run: {done} samples, total time: {elapsed:.1f}s, Failures: {_fail_count}")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
