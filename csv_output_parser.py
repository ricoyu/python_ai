import os

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions= "你的响应应该是csv格式的逗号分隔值的列表, 例如 '内容1, 内容2, 内容3'"

prompt = PromptTemplate(template="{format_instructions}\n请列出5个{subject}", input_variables=["subject"],
                          partial_variables={"format_instructions": format_instructions})

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url)

parser = output_parser
chain = prompt | model | parser

result = chain.invoke({"subject": "冰激凌口味"})

print(result) # ['巧克力', '香草', '草莓', '薄荷巧克力', '抹茶']
