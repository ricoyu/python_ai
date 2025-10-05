import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(model="gpt-3.5-turbo",
    base_url=api_base,
    openai_api_key=api_key)

message = model.invoke([
    HumanMessage(content="请根据下面的主题写一篇小红书营销短文: 康师傅绿茶")
])
# message = model.invoke([
#     {"role": "user", "content": "请根据下面的主题写一篇小红书营销短文: 康师傅绿茶"}
# ])
print(message.content) # 类型是 langchain_core.messages.ai.AIMessage