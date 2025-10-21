import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.3)


# 定义你想要的数据结构
class Book(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    description: str = Field(description="书的简介")


# 以及旨在提示语言模型填充数据结构的查询
query = "请给我介绍学习中国历史的经典书籍"

output_parser = JsonOutputParser(pydantic_object=Book)

format_instructions = output_parser.get_format_instructions()
format_instructions = '''输出应格式化为符合以下 JSON 结构的 JSON 实例。
JSON结构
```
{{
"title": "书的标题",
"author": "作者",
"description": "书的简介"
}}
```
'''

prompt = PromptTemplate(template=f"{format_instructions}\n{query}\n", input_variables=["query"],
                        partial_variables={"format_instructions": format_instructions}, )

chain = prompt | model | output_parser
output = chain.invoke({"query": query})
print(output)