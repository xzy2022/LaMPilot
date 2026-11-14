"""GPT / OpenAI 接入适配，支持 gpt-6.1-sol 推理模型（low reasoning effort）与标准 Chat 模型。
复用官方提示词、消息构造和 Python 解析器。
"""

import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from projects.lampilot.dt.cg_agent import CodeGenerationAgent


class ReasoningChatOpenAI(ChatOpenAI):
    """适配 OpenAI 推理模型（如 gpt-6.1-sol、o1、o3 等），必须剔除 temperature 参数以防 API 报错。"""
    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        params.pop("temperature", None)
        return params


class GPTAgent(CodeGenerationAgent):
    def __init__(self, model_name="gpt-6.1-sol", zero_shot=False):
        api_key = (os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("请先设置 GPT_API_KEY 或 OPENAI_API_KEY；不要把密钥写入代码或 Git。")

        base_url = (os.environ.get("GPT_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip() or None

        # 模型别名规范化
        canonical_model = model_name
        if model_name.lower().replace("-", "") == "gpt6.1sol":
            canonical_model = "gpt-6.1-sol"

        is_reasoning_model = any(
            x in canonical_model.lower() for x in ("gpt-6.1-sol", "gpt6.1sol", "o1", "o3")
        )

        if is_reasoning_model:
            # gpt-6.1-sol 运行于 low reasoning effort 模式，不传 temperature
            self.llm = ReasoningChatOpenAI(
                model=canonical_model,
                api_key=api_key,
                base_url=base_url,
                timeout=120,
                max_retries=1,
                reasoning_effort="low",
            )
        else:
            self.llm = ChatOpenAI(
                model=canonical_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                timeout=120,
                max_retries=1,
            )

        self.zero_shot = zero_shot
        self.last_response = {}

    def step(self):
        message = self.llm.invoke(self.messages)
        metadata = message.response_metadata
        usage = metadata.get("token_usage", {})
        self.last_response = {
            "text": message.content,
            "usage": usage,
            "model": metadata.get("model_name"),
            "finish_reason": metadata.get("finish_reason"),
            "system_fingerprint": metadata.get("system_fingerprint"),
        }
        if metadata.get("finish_reason") == "length":
            raise RuntimeError("模型输出达到 token 上限，未执行可能截断的策略。")
        parsed = self.process_ai_message(message)
        if not parsed["program_code"]:
            raise RuntimeError("官方解析器没有提取到有效 Python 策略函数。")
        self.code.update({
            "program_code": parsed["program_code"],
            "program_name": parsed["program_name"],
        })
        self.conversations.append((
            self.messages[0].content, self.messages[1].content, message.content,
        ))
        return {
            "reused_code": "",
            "new_code": parsed["program_code"] + "\n" + parsed["exec_code"],
        }
