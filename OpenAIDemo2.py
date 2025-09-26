import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销短文: {topic}")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

message = chain.invoke({"topic": "康师傅绿茶"})
print(message)
