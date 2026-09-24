"""下载并预处理所有数据集，输出到 data/processed/。

用法：
    python scripts/01_download_data.py
"""
import sys
from pathlib import Path

# 把项目根加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.prepare import (
    load_clinc150, load_mixatis, load_mixsnips,
    split_calibration, build_knowledge_base, save_jsonl,
)
import yaml

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
OUT = ROOT / "data" / "processed"


def main():
    # CLINC150
    print("=== CLINC150 ===")
    clinc = load_clinc150(oos_heldout=CFG["data"]["oos_heldout_intents"], seed=CFG["seed"])
    train, calib = split_calibration(clinc["train"], CFG["data"]["calibration_ratio"], CFG["seed"])
    save_jsonl(train, OUT / "clinc150" / "train.jsonl")
    save_jsonl(calib, OUT / "clinc150" / "calib.jsonl")
    save_jsonl(clinc["val"], OUT / "clinc150" / "val.jsonl")
    save_jsonl(clinc["test"], OUT / "clinc150" / "test.jsonl")
    kb = build_knowledge_base(train, clinc["in_domain_intents"],
                              CFG["data"]["max_examples_per_intent"])
    import json
    json.dump(kb, open(OUT / "clinc150" / "kb.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  train={len(train)} calib={len(calib)} val={len(clinc['val'])} test={len(clinc['test'])}")
    print(f"  in-domain intents={len(clinc['in_domain_intents'])}")

    # MixATIS
    print("=== MixATIS ===")
    ma = load_mixatis()
    train_ma, calib_ma = split_calibration(ma["train"], CFG["data"]["calibration_ratio"], CFG["seed"])
    save_jsonl(train_ma, OUT / "mixatis" / "train.jsonl")
    save_jsonl(calib_ma, OUT / "mixatis" / "calib.jsonl")
    save_jsonl(ma["test"], OUT / "mixatis" / "test.jsonl")
    kb_ma = build_knowledge_base(train_ma, ma["in_domain_intents"],
                                  CFG["data"]["max_examples_per_intent"])
    json.dump(kb_ma, open(OUT / "mixatis" / "kb.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  train={len(train_ma)} calib={len(calib_ma)} test={len(ma['test'])}")

    # MixSNIPS
    print("=== MixSNIPS ===")
    ms = load_mixsnips()
    train_ms, calib_ms = split_calibration(ms["train"], CFG["data"]["calibration_ratio"], CFG["seed"])
    save_jsonl(train_ms, OUT / "mixsnips" / "train.jsonl")
    save_jsonl(calib_ms, OUT / "mixsnips" / "calib.jsonl")
    save_jsonl(ms["test"], OUT / "mixsnips" / "test.jsonl")
    kb_ms = build_knowledge_base(train_ms, ms["in_domain_intents"],
                                  CFG["data"]["max_examples_per_intent"])
    json.dump(kb_ms, open(OUT / "mixsnips" / "kb.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  train={len(train_ms)} calib={len(calib_ms)} test={len(ms['test'])}")

    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
