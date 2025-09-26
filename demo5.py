# 提示模板
from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-4-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    # openai_api_key="sk-zk21423fce4ce26232550b939c668a7732b6895db76b7859",  # 智增增的KEY
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

search = TavilySearchResults(max_results=2) # 返回值只有两个结果
results = search.invoke("北京天气怎么样?")
for result in results:
    print(result)

resp = model.invoke([HumanMessage(content="北京天气怎么样?")])
print(resp.content)

# 当你使用model.bind_tools([search])时，你只是告诉模型 "你有权限使用这个搜索工具"，但并没有实际执行搜索的能力。
model_bind_tools = model.bind_tools([search])
resp = model_bind_tools.invoke([HumanMessage(content="中国的首都是哪个城市?")])  # 这时候还不会真正调工具
print(f"Model result: {resp.content}")
print(f"Tools result: {resp.tool_calls}")
resp2 = model_bind_tools.invoke([HumanMessage(content="北京天气怎么样?")])
print(f"Model result: {resp2.content}")
print(f"Tools result: {resp2.tool_calls}")