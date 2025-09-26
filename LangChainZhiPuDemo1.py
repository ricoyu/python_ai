"""
LangChain封装智谱AI
"""
from typing import List, Dict, Optional

from langchain_core.language_models import LLM
import os

from langchain_core.messages import AIMessage
from zhipuai import ZhipuAI


class ChatGLM4(LLM):
    history: List[Dict[str, str]] = []
    client: Optional[ZhipuAI] = None

    def __init__(self):
        super().__init__()
        api_key = os.getenv("ZHIPU_API_KEY")
        self.client = ZhipuAI(api_key=api_key)

    @property
    def _llm_type(selfself):
        return "ChatGLM4"

    def invoke(self, prompt, history=[]):
        if history is None:
            self.history = []
        self.history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model="glm-4", messages=self.history)
        result = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": result})
        return AIMessage(content=result)

    def _call(self, prompt, history=[]):
        return self.invoke(prompt, self.history)

model = ChatGLM4()
message = model.invoke("中国的首都是哪里?")
print(message.content)
message = model.invoke("这个城市有哪些美食?")
print(message.content)

