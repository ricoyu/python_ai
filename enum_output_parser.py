import os
from enum import Enum

from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class Colors(Enum):
    RED = "红色"
    YELLOW = "黄色"
    BLACK = "黑色"
    WHITE = "白色"
    BROWN = "棕色"


output_parser = EnumOutputParser(enum=Colors)

instructions = output_parser.get_format_instructions()
print(instructions)
instructions = """响应结果请选择以下选项之一:红色 黄色 黑色 白色 棕色。 不要有其他的内容"""
template = PromptTemplate.from_template("""
{person}的皮肤主要是什么颜色?

{instructions}""")

promptTemplate = PromptTemplate.from_template("""
{person}的皮肤主要是什么颜色?

{instructions}""")
api_key = os.getenv("ZZZ_API_KEY")
api_bsae = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=api_bsae)

prompt = promptTemplate.partial(instructions=instructions)
chain = prompt | model | output_parser

output = chain.invoke({"person": "亚洲人"})
print(output)  # <Colors.YELLOW: '黄色'>