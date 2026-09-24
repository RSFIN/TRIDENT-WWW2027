"""为每个数据集构建 FAISS 检索索引。

用法：
    python scripts/02_build_index.py
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
DATASETS = ["clinc150", "mixatis", "mixsnips"]


def main():
    embedder = Embedder(CFG["retriever"]["embed_model"])
    for ds in DATASETS:
        kb_path = ROOT / "data" / "processed" / ds / "kb.json"
        if not kb_path.exists():
            print(f"[skip] {ds}: kb.json not found, run 01 first")
            continue
        kb = json.load(open(kb_path, encoding="utf-8"))
        print(f"=== {ds}: building index, {len(kb)} intents ===")
        idx = IntentIndex(embedder, kb)
        out = ROOT / "data" / "indexes" / ds
        idx.save(str(out))
        print(f"  saved to {out}")


if __name__ == "__main__":
    main()
