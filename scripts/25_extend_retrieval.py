"""Extend the retrieval candidates of existing predictions to top-10 (local embedding retrieval, no LLM API).

用法：
    python scripts/25_extend_retrieval.py --dataset clinc150
    python scripts/25_extend_retrieval.py --dataset mixatis
    python scripts/25_extend_retrieval.py --dataset mixsnips

读取 data/predictions/{ds}/*.jsonl（retrieved=top-5，llm_* 已生成），
重检索 top-10 覆盖 retrieved 字段，写入 data/predictions/{ds}_k10/。
LLM 输出（llm_main/llm_aux/llm_confidence）原样保留；OOS 记录保持 retrieved=[]。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from src.retriever.embedder import Embedder
from src.retriever.index import IntentIndex

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
TOP_K = 10


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset

    embedder = Embedder(CFG["retriever"]["embed_model"])
    kb = json.load(open(ROOT / "data" / "processed" / ds / "kb.json", encoding="utf-8"))
    idx = IntentIndex.load(str(ROOT / "data" / "indexes" / ds), embedder, kb)

    src_dir = ROOT / "data" / "predictions" / ds
    dst_dir = ROOT / "data" / "predictions" / f"{ds}_k10"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in ("calib", "test"):
        src = src_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        n = 0
        with open(src, encoding="utf-8") as fin, \
             open(dst_dir / f"{split}.jsonl", "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("oos"):
                    cands = idx.search(rec["utterance"], top_k=TOP_K)
                    rec["retrieved"] = [(it, sim) for it, sim in cands]
                    rec["top1_sim"] = cands[0][1] if cands else 0.0
                # oos 记录保持 retrieved=[]；llm_* 原样保留
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"[{ds}/{split}] {n} records -> {dst_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
