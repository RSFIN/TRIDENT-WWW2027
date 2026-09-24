"""DashScope OpenAI 兼容 client。"""
import os
import re
import json
from typing import Dict, List, Optional
from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = None, temperature: float = 0.0, max_tokens: int = 256):
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                     "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model = model or os.environ.get("LLM_MODEL", "qwen-plus")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    def classify_intent(self, query: str, candidates: List[Dict]) -> Dict:
        """输入查询 + 检索到的候选意图，输出主意图、次意图、自报置信。

        candidates: [{"intent": str, "definition": str, "examples": [str]}]
        返回: {"main": str, "aux": [str], "confidence": {intent: float}}
        """
        cand_text = "\n".join(
            f"- {c['intent']}: {c['definition']}（示例：{'; '.join(c['examples'][:2])}）"
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

    @staticmethod
    def _parse_json(text: str) -> Dict:
        # 提取 JSON 块
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"main": "", "aux": [], "confidence": {}}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"main": "", "aux": [], "confidence": {}}
