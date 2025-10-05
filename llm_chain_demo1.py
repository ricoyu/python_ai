import os

from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(openai_api_base="https://api.zhizengzeng.com/v1",
                openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1")

chat_template = ChatPromptTemplate.from_template("你是一只小猫, 你的名字叫做{name}, 今天{user_input}")
llm_dodel = LLMChain(llm=model, prompt=chat_template)
response = llm_dodel.run(name="小猫", user_input="你扑我了吗?") #返回的已经是字符串了
print(response)
