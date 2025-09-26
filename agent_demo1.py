"""
Agent代理的使用
"""
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

model = ChatOpenAI(
    model_name="gpt-4-turbo",
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

# Langchain内置了一个工具, 可以轻松地使用Tavily搜索引擎作为工具
search = TavilySearchResults(max_results=2)
# print(search.invoke("苏州的天气怎么样?"))

# 让模型绑定工具
tools = [search]

# model_with_tools = model.bind_tools(tools)
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
resp = agent_executor.invoke({"messages": [HumanMessage(content="中国的首都是哪个城市?")]})
print(resp["messages"])

resp2 = agent_executor.invoke({"messages": [HumanMessage(content="苏州天气怎么样?")]})
print(resp2["messages"][3].content)
