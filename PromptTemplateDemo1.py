import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")
model = ChatOpenAI(api_key=api_key, base_url=api_base)
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销短文: {topic}")
prompt_value = prompt.invoke({"topic": "康师傅绿茶"})
print(prompt_value) #messages=[HumanMessage(content='请根据下面的主题写一篇小红书营销短文: 康师傅绿茶', additional_kwargs={}, response_metadata={})]

message = model.invoke(prompt_value) # AIMessage类型
print(message)

output_parser = StrOutputParser()
parser_value = output_parser.invoke(message)
print(parser_value) #str类型

