import os

from langchain_openai import ChatOpenAI

# OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")
api_key = os.getenv("ZZZ_API_KEY")

llm = ChatOpenAI(openai_api_key=api_key,
                 base_url=base_url)
base_message = llm.invoke("中国的首都是哪里, 不需要介绍") #是一个AIMessage类型
print(base_message.content)
