"""DeepSeek 接入适配，复用官方提示词、消息构造和 Python 解析器。"""

import os

from langchain_openai import ChatOpenAI

from projects.lampilot.dt.cg_agent import CodeGenerationAgent


class DeepSeekAgent(CodeGenerationAgent):
    def __init__(self, model_name="deepseek-flash", zero_shot=False):
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("请先设置 DEEPSEEK_API_KEY；不要把密钥写入代码或 Git。")
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.0,
            timeout=90,
            max_retries=1,
            max_tokens=2048,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self.zero_shot = zero_shot
        self.last_response = {}

    def step(self):
        message = self.llm.invoke(self.messages)
        metadata = message.response_metadata
        self.last_response = {
            "text": message.content,
            "usage": metadata.get("token_usage", {}),
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
