"""嵌入模型封装（CPU 本地）。"""
import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None):
        if device is None:
            device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True,
        )
        return emb.astype(np.float32)
