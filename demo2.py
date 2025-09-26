import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"

# 这个是LangSmith的KEY
os.environ["LANGCHAIN_API_KEY"] = "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"

# 调用大预言模型
# model = ChatOpenAI(model="gpt-3.5-turbo")
# 关键参数修改：指定DeepSeek的API地址和密钥
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024   # 可根据需要调整
)

msg = [
    {"role": "system", "content": "请将一下内容翻译成英文"},
    {"role": "user", "content": "你好吗请问你要去哪里"}
]

parser = StrOutputParser()

chain = model | parser

result = chain.invoke(msg)
print( result)