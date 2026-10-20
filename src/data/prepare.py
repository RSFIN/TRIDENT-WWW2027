"""下载并预处理公开意图数据集，统一为 IntentSample schema。

数据源：
- CLINC150: HuggingFace datasets (clinc/clinc_oos)
- MixATIS:  PolyAI/mixatis
- MixSNIPS: PolyAI/mixsnips
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from .schema import IntentSample

ROOT = Path(__file__).resolve().parents[2]


def load_clinc150(oos_heldout: int = 15, seed: int = 42) -> Dict[str, List[IntentSample]]:
    """CLINC150：150 意图，留 oos_heldout 个意图作 OOS。"""
    ds = load_dataset("clinc/clinc_oos", "plus")
    all_names = ds["train"].features["intent"].names
    oos_name = "oos" if "oos" in all_names else all_names[-1]
    intent_names = [n for n in all_names if n != oos_name]
    rng = random.Random(seed)
    held_out = set(rng.sample(intent_names, oos_heldout))
    in_domain = [n for n in intent_names if n not in held_out]

    def split(split_ds):
        samples = []
        for row in split_ds:
            label_id = row["intent"]
            intent_name = all_names[label_id]
            if intent_name == oos_name or intent_name in held_out:
                samples.append(IntentSample(
                    utterance=row["text"], intents=[], main_intent="",
                    oos=True, dataset="clinc150",
                ))
            else:
                samples.append(IntentSample(
                    utterance=row["text"], intents=[intent_name],
                    main_intent=intent_name, oos=False, dataset="clinc150",
                ))
        return samples

    return {
        "train": split(ds["train"]),
        "val": split(ds["validation"]),
        "test": split(ds["test"]),
        "in_domain_intents": in_domain,
    }


def _download_text(url: str, cache_path: str) -> str:
    """下载文本文件到本地缓存，返回内容。"""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(cache_path).exists():
        return open(cache_path, encoding="utf-8").read()
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read().decode("utf-8")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def _parse_agif_txt(content: str, dataset: str) -> List[IntentSample]:
    """解析 AGIF BIO 格式：
    每个句子是一块：前几行是 'token\\tslot'，最后一行是 'intent1#intent2#...'，块间空行。
    """
    samples = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = [l for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue
        # 最后一行是 intent labels（不含 tab）
        intent_line = lines[-1]
        if "\t" in intent_line:
            continue  # 格式不对，跳过
        tokens = []
        for line in lines[:-1]:
            parts = line.split("\t")
            tokens.append(parts[0])
        if not tokens:
            continue
        utterance = " ".join(tokens).strip()
        labels = [l for l in intent_line.split("#") if l]
        if not labels:
            continue
        samples.append(IntentSample(
            utterance=utterance, intents=labels,
            main_intent=labels[0], oos=False, dataset=dataset,
        ))
    return samples


def load_mixatis(data_dir: str = "data/raw/MixATIS_clean") -> Dict[str, List[IntentSample]]:
    """MixATIS：从 AGIF repo 的 txt 文件加载。"""
    base = "https://ghfast.top/https://raw.githubusercontent.com/LooperXX/AGIF/master/data/MixATIS_clean"
    cache = ROOT / data_dir
    train = _parse_agif_txt(
        _download_text(f"{base}/train.txt", cache / "train.txt"), "mixatis")
    dev = _parse_agif_txt(
        _download_text(f"{base}/dev.txt", cache / "dev.txt"), "mixatis")
    test = _parse_agif_txt(
        _download_text(f"{base}/test.txt", cache / "test.txt"), "mixatis")
    all_intents = sorted({i for s in train for i in s.intents})
    return {"train": train, "val": dev, "test": test,
            "in_domain_intents": all_intents}


def load_mixsnips(data_dir: str = "data/raw/MixSNIPS_clean") -> Dict[str, List[IntentSample]]:
    """MixSNIPS：从 AGIF repo 的 txt 文件加载。"""
    base = "https://ghfast.top/https://raw.githubusercontent.com/LooperXX/AGIF/master/data/MixSNIPS_clean"
    cache = ROOT / data_dir
    train = _parse_agif_txt(
        _download_text(f"{base}/train.txt", cache / "train.txt"), "mixsnips")
    dev = _parse_agif_txt(
        _download_text(f"{base}/dev.txt", cache / "dev.txt"), "mixsnips")
    test = _parse_agif_txt(
        _download_text(f"{base}/test.txt", cache / "test.txt"), "mixsnips")
    all_intents = sorted({i for s in train for i in s.intents})
    return {"train": train, "val": dev, "test": test,
            "in_domain_intents": all_intents}


def split_calibration(train: List[IntentSample], ratio: float = 0.25, seed: int = 42):
    """从 train 切 calibration，保持意图分布。"""
    rng = random.Random(seed)
    by_intent: Dict[str, List[IntentSample]] = {}
    for s in train:
        if s.oos:
            by_intent.setdefault("__oos__", []).append(s)
        else:
            by_intent.setdefault(s.main_intent, []).append(s)

    calib, new_train = [], []
    for intent, samples in by_intent.items():
        rng.shuffle(samples)
        n_cal = max(1, int(len(samples) * ratio))
        calib.extend(samples[:n_cal])
        new_train.extend(samples[n_cal:])
    rng.shuffle(calib)
    rng.shuffle(new_train)
    return new_train, calib


def build_knowledge_base(train: List[IntentSample], intents: List[str],
                         max_examples: int = 10) -> Dict[str, Dict]:
    """从 train 构建意图知识库：{intent: {definition, examples, slots}}。

    当前用示例话语作 definition 占位；后续可 LLM 生成正式定义。
    """
    kb = {}
    by_intent: Dict[str, List[str]] = {i: [] for i in intents}
    for s in train:
        if not s.oos:
            for intent in s.intents:
                if intent in by_intent:
                    by_intent[intent].append(s.utterance)
    for intent in intents:
        examples = by_intent[intent][:max_examples]
        kb[intent] = {
            "definition": intent.replace("_", " "),  # 占位
            "examples": examples,
            "num_examples": len(examples),
        }
    return kb


def save_jsonl(samples: List[IntentSample], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[IntentSample]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(IntentSample.from_dict(json.loads(line)))
    return out
