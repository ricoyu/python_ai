import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base=api_base,  # 智增增的DE BASE_URL
    openai_api_key=api_key,  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024   # 可根据需要调整
)

msg = [
    {"role": "system", "content": "你是一名中国的资深导游, 对中国各个城市的历史人文都非常了解, 也非常了解各个城市的特色美食"},
    {"role": "user", "content": "中国的首都是哪里?"}
]

parser = StrOutputParser()

chain = model | parser

answer = chain.invoke(msg)
print(answer)
msg.append({"role": "assistant", "content": answer})
msg.append({"role": "user", "content": "这个城市什么景点最有名?"})
answer = chain.invoke(msg)
print(answer)
msg.append({"role": "assistant", "content": answer})
msg.append({"role": "user", "content": "这个城市有什么特色美食?"})
answer = chain.invoke(msg)
print(answer)
