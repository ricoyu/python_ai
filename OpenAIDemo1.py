import os

from langchain_openai import ChatOpenAI

# model = ChatOpenAI()

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
base_message = llm.invoke("中国的首都是哪里, 不需要介绍")
