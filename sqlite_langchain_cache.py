import os
import time

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=api_base)

prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销短文: {topic}")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

start_time = time.time()
set_llm_cache(SQLiteCache(database_path="./db/langchain.db"))
result = chain.invoke({"topic": "旺仔小馒头"})
end_time = time.time()
print(f"总共花费了{end_time - start_time}秒")  # 总共花费了7.156178712844849秒
# print( result)


start_time = time.time()
result = chain.invoke({"topic": "旺仔小馒头"})
end_time = time.time()
print(f"总共花费了{end_time - start_time}秒")  # 总共花费了0.0019991397857666016秒
# print( result)