"""统一数据 schema。

每条样本：
{
  "utterance": str,
  "intents": List[str],   # 所有意图标签，顺序无所谓
  "main_intent": str,    # 主意图标签（数据集标注的第一个）
  "oos": bool,           # 是否域外
  "dataset": str,        # 来源数据集
}
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class IntentSample:
    utterance: str
    intents: List[str]
    main_intent: str
    oos: bool = False
    dataset: str = ""

    def to_dict(self):
        return {
            "utterance": self.utterance,
            "intents": self.intents,
            "main_intent": self.main_intent,
            "oos": self.oos,
            "dataset": self.dataset,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
