import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销短文: {topic}")
model = ChatOpenAI(api_key=api_key, base_url=api_base)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

message = chain.invoke({"topic": "康师傅绿茶"})
print(message)
