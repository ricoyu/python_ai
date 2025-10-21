import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI

human_prompt = "用 {word_count} 字总结我们迄今为止的对话"

human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversion"), human_message_template])

human_message = HumanMessage(content="学习编程最好的方法是什么?")
ai_message = AIMessage(content="""
1. 选择编程语言: 决定你想要学习的编程语言.
2. 从基础开始, 熟悉变量, 数据类型和控制结构等基本编程概念.
3. 练习, 练习, 再练习:学习编程最好的方法是通过实践经验
""")

messages = chat_prompt.format_prompt(conversion=[human_message, ai_message], word_count="10").to_messages()
# for msg in messages:
    # print(type(msg))
    # print(msg)

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(
    base_url= base_url,
    api_key=api_key
)

output_parser = StrOutputParser()

chain = chat_prompt | model | output_parser

result = chain.invoke({"conversion": [human_message, ai_message], "word_count": "10"})
print(result)
