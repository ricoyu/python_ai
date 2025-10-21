import os

from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = DatetimeOutputParser()
template = """
回答用户的问题:

{question}

{format_instructions}"""

format_instructions = output_parser.get_format_instructions()
# format_instructions 是英文的如下这一串
"""
Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.
Examples: 2023-07-04T14:30:00.000000Z, 1999-12-31T23:59:59.999999Z, 2025-01-01T00:00:00.000000Z
Return ONLY this string, no other words!
"""

# 下面是为了改用中文提示词
format_instructions = ''' 响应的格式用日期时间字符串: "% Y-% m-% dT% H:% M:% S.% fZ"。
示例: 1898-05-31T06:59:40.248940Z, 1808-10-20T01:56:09.167633Z、0226-10-17T06:18:24.192024Z

仅返回此字符串，没有其他单词！'''

prompt = PromptTemplate.from_template(template, partial_variables={"format_instructions": format_instructions}, )

api_key = os.getenv("ZZZ_API_KEY")
api_base = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=api_base, )

chain = prompt | model | output_parser

result = chain.invoke({"question": "比特币是什么时候创立的?"})
print(result)  # datetime.datetime(2009, 1, 3, 18, 15, 5)