"""DashScope OpenAI 兼容 client（带超时与重试，防 API 挂起）。"""
import os
import re
import json
import time
from typing import Dict, List, Optional
from openai import OpenAI

RETRY_TIMES = 3
TIMEOUT = 120.0


class LLMClient:
    def __init__(self, model: str = None, temperature: float = 0.0, max_tokens: int = 256):
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                     "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=TIMEOUT,
            max_retries=0,  # 由本类自行重试，便于观测
        )
        self.model = model or os.environ.get("LLM_MODEL", "qwen-plus")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str) -> str:
        last_err = None
        for attempt in range(1, RETRY_TIMES + 1):
            try:
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
            except Exception as e:  # 网络/超时/限流均重试
                last_err = e
                wait = 5 * (2 ** (attempt - 1))
                print(f"    [llm retry {attempt}/{RETRY_TIMES}] {type(e).__name__}: {e} (wait {wait}s)")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {RETRY_TIMES} retries: {last_err}")

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
