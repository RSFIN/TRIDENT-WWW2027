"""对 train/calib/test 跑 LLM 推理，缓存结果到 data/predictions/。

用法：
    python scripts/03_run_llm.py --dataset clinc150 --split test [--workers 6]

并发说明：LLM 调用彼此独立（温度 0 确定性），--workers>1 用线程池并行。
检索（embedder 编码）必须先串行完成（多线程编码会死锁），检索结果缓存后
再并发调用 LLM；结果按样本原始顺序写回，不改变实验结果语义。
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
from src.retriever.index import IntentIndex
from src.llm.client import LLMClient

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))

_fail_count = 0  # 模块级并发计数器


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "calib", "test"])
    ap.add_argument("--limit", type=int, default=None, help="调试时限制样本数")
    ap.add_argument("--out-subdir", type=str, default="",
                    help="预测输出目录后缀：data/predictions/{ds}[_suffix]；默认空=主目录")
    ap.add_argument("--workers", type=int, default=1,
                    help="LLM 并行线程数（默认 1=串行，与历史结果一致）")
    args = ap.parse_args()

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / args.dataset / "kb.json", encoding="utf-8"))
    idx = IntentIndex.load(str(ROOT / "data" / "indexes" / args.dataset), embedder, kb)
    llm = LLMClient(model=CFG["llm"]["model"])

    samples = load_jsonl(ROOT / "data" / "processed" / args.dataset / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]

    out_dir = ROOT / "data" / "predictions" / \
        (args.dataset if not args.out_subdir else f"{args.dataset}_{args.out_subdir}")
    out_path = out_dir / f"{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：读取已完成的 utterance，跳过
    done = set()
    if out_path.exists():
        for line in open(out_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["utterance"])
            except Exception:
                pass
    if done:
        print(f"[resume] {len(done)} samples already done, skipping")

    todo = [(i, s) for i, s in enumerate(samples) if s.utterance not in done]
    print(f"[todo] {len(todo)} samples to process (workers={args.workers})")

    # ---- Phase 1: 串行检索（embedder 编码不能并发）----
    results = {}        # index -> rec（含 OOS 直接完成的）
    llm_todo = []       # (index, sample, cands, top1_sim)
    t0 = time.time()
    for i, s in todo:
        if s.oos:
            top1 = idx.top1_sim(s.utterance)
            results[i] = {
                "utterance": s.utterance, "intents": s.intents,
                "main_intent": s.main_intent, "oos": True,
                "top1_sim": top1,
                "llm_main": "", "llm_aux": [], "llm_confidence": {},
                "retrieved": [],
            }
            continue
        cands = idx.search(s.utterance, top_k=CFG["retriever"]["top_k_r"])
        top1 = cands[0][1] if cands else 0.0
        llm_todo.append((i, s, cands, top1))
    print(f"[retrieval] {len(llm_todo)} in-domain in {time.time()-t0:.0f}s")

    # ---- Phase 2: 并发 LLM ----
    lock = __import__("threading").Lock()

    def llm_only(item):
        global _fail_count
        i, s, cands, top1 = item
        cand_dicts = [
            {"intent": it, "definition": kb[it]["definition"],
             "examples": kb[it]["examples"]}
            for it, _ in cands
        ]
        try:
            result = llm.classify_intent(s.utterance, cand_dicts)
        except Exception as e:
            with lock:
                _fail_count += 1
                nf = _fail_count
            print(f"    [FAIL {nf}] {s.utterance[:60]!r}: {e}")
            result = {"main": "", "aux": [], "confidence": {}}
        return i, {
            "utterance": s.utterance, "intents": s.intents,
            "main_intent": s.main_intent, "oos": False,
            "top1_sim": top1,
            "retrieved": [(it, sim) for it, sim in cands],
            "llm_main": result.get("main", ""),
            "llm_aux": result.get("aux", []),
            "llm_confidence": result.get("confidence", {}),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(llm_only, item) for item in llm_todo]
        n_done = 0
        for fut in as_completed(futures):
            i, rec = fut.result()
            results[i] = rec
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [llm] {n_done}/{len(llm_todo)}", flush=True)

    # ---- Phase 3: 按原始顺序写回 ----
    with open(out_path, "a", encoding="utf-8") as fout:
        written = 0
        for i, s in enumerate(samples):
            if s.utterance in done:
                continue
            fout.write(json.dumps(results[i], ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                print(f"  [{args.dataset}/{args.split}] wrote {written}/{len(todo)}")
                fout.flush()
    print(f"Done: {out_path} (llm failures={_fail_count})")


if __name__ == "__main__":
    main()
