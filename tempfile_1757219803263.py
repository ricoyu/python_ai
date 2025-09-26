# temperature.py
import os
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage

# 设置API密钥
os.environ["TAVILY_API_KEY"] = "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"

# 初始化模型
model = ChatOpenAI(
    model_name="gpt-4-turbo",
    openai_api_base="https://api.zhizengzeng.com/v1",
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",
    temperature=0.7,
    max_tokens=1024
)

# 创建搜索工具
search = TavilySearchResults(max_results=2)

# 创建工具列表
tools = [search]

# 创建代理执行器
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

def query_weather(city):
    """
    查询指定城市的天气信息
    
    Args:
        city (str): 城市名称
    
    Returns:
        str: 天气信息
    """
    query = f"{city} 天气怎么样?"
    response = agent_executor.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    # 提取并返回结果
    return response["messages"][-1].content

if __name__ == "__main__":
    # 示例使用
    print("天气查询助手已启动")
    
    # 查询北京天气
    result = query_weather("北京")
    print(f"北京天气: {result}")
    
    # 查询上海天气
    result = query_weather("上海")
    print(f"上海天气: {result}")
    
    # 交互式查询
    while True:
        city = input("\n请输入要查询的城市名称 (输入 'quit' 退出): ")
        if city.lower() == 'quit':
            break
        try:
            result = query_weather(city)
            print(f"{city}天气: {result}")
        except Exception as e:
            print(f"查询出错: {e}")
