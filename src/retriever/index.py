"""FAISS 检索索引：意图级原型 + 示例级向量。"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import faiss

from .embedder import Embedder


class IntentIndex:
    def __init__(self, embedder: Embedder, kb: Dict[str, Dict]):
        self.embedder = embedder
        self.kb = kb
        self.intents = list(kb.keys())
        self._build()

    def _build(self):
        # 意图级原型 = 所有示例向量的均值
        self.intent_texts = []
        self.intent_labels = []
        for intent, info in self.kb.items():
            for ex in info["examples"]:
                self.intent_texts.append(ex)
                self.intent_labels.append(intent)

        # 示例级向量
        self.example_emb = self.embedder.encode(self.intent_texts)
        self.example_index = faiss.IndexFlatIP(self.example_emb.shape[1])
        self.example_index.add(self.example_emb)

        # 意图级原型向量
        self.prototype_emb = np.zeros((len(self.intents), self.example_emb.shape[1]), dtype=np.float32)
        for i, intent in enumerate(self.intents):
            mask = np.array([l == intent for l in self.intent_labels])
            if mask.sum() > 0:
                self.prototype_emb[i] = self.example_emb[mask].mean(axis=0)
        self.prototype_index = faiss.IndexFlatIP(self.prototype_emb.shape[1])
        self.prototype_index.add(self.prototype_emb)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """返回 [(intent, cosine_sim)]，按相似度降序。"""
        q_emb = self.embedder.encode([query])
        # 用示例级检索，聚合到意图
        D, I = self.example_index.search(q_emb, top_k * 3)
        scores: Dict[str, float] = {}
        for d, idx in zip(D[0], I[0]):
            intent = self.intent_labels[idx]
            scores[intent] = max(scores.get(intent, -1), float(d))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def top1_sim(self, query: str) -> float:
        """OOS gate 用：Top1 相似度。"""
        q_emb = self.embedder.encode([query])
        D, _ = self.prototype_index.search(q_emb, 1)
        return float(D[0][0])

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.example_index, f"{path}/example.faiss")
        faiss.write_index(self.prototype_index, f"{path}/prototype.faiss")
        np.save(f"{path}/example_emb.npy", self.example_emb)
        with open(f"{path}/meta.json", "w", encoding="utf-8") as f:
            json.dump({"intents": self.intents, "labels": self.intent_labels}, f)

    @classmethod
    def load(cls, path: str, embedder: Embedder, kb: Dict[str, Dict]):
        obj = cls.__new__(cls)
        obj.embedder = embedder
        obj.kb = kb
        with open(f"{path}/meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj.intents = meta["intents"]
        obj.intent_labels = meta["labels"]
        obj.example_emb = np.load(f"{path}/example_emb.npy")
        obj.example_index = faiss.read_index(f"{path}/example.faiss")
        obj.prototype_index = faiss.read_index(f"{path}/prototype.faiss")
        return obj
