"""
对话提示词
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

#要包含在最终提示词中得字典示例列表
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2的平方式多少?", "output": "4"},
    {"input": "2+3", "output": "5"}
]


#组成少示例的提示词模板
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
"""
Human: 2+2
AI: 4
Human: 2+3
AI: 5"""
print(few_shot_prompt.format())

#最后, 组装最终的提示并将其与模型一起使用
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是以为厉害的数学天才"),
    few_shot_prompt,
    ("human", "{input}")
])

model = ChatOpenAI(api_key=api_key, base_url=api_base)

parser = StrOutputParser()

chain = final_prompt | model | parser

response = chain.invoke({"input": "3的平方式多少?"})
print(response)
