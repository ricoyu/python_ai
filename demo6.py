# 提示模板
import os

from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

os.environ["TAVILY_API_KEY"] = "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"
model = ChatOpenAI(
    model_name="gpt-4-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    # openai_api_key="sk-zk21423fce4ce26232550b939c668a7732b6895db76b7859",  # 智增增的KEY
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

search = TavilySearchResults(max_results=2) # 返回值只有两个结果

# 当你使用model.bind_tools([search])时，你只是告诉模型 "你有权限使用这个搜索工具"，但并没有实际执行搜索的能力。
tools = [search]

# 模型能够推理是否需要调用工具去完成用户的答案
# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
resp = agent_executor.invoke({"messages": [HumanMessage(content="中国的首都是哪个城市?")]})
print(f"Model result: {resp["messages"][1].content}")
resp2 = agent_executor.invoke({"messages": [HumanMessage(content="北京天气怎么样?")]})
print(f"agent result: {resp2["messages"][3].content}")

# 常见问题检查点：
# 1. 语法错误检查
# 2. 变量命名规范
# 3. 函数定义和调用
# 4. 异常处理
# 5. 资源管理（文件、网络连接等）
# 6. 代码逻辑正确性

# 示例问题修复区域：
# 如果有以下问题需要修复：

# 问题1: 可能缺少必要的导入
# import os
# import sys

# 问题2: 函数缺少异常处理
# try:
#     # ... existing code...
# except Exception as e:
#     print(f"发生错误: {e}")

# 问题3: 变量作用域问题
# def some_function():
#     global some_variable
#     # ... existing code...
