import os

from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = XMLOutputParser()

format_instructions = """
响应以xml的结构返回, 使用如下xml结构:
<xm l>
    <movie>电影1</movie>
    <movie>电影2</movie>
</xml>
"""

prompt = PromptTemplate(template="""{query}\n{format_instructions}""",
                        input_variables=["query"],
                        partial_variables={"format_instructions": format_instructions},
                        )

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.3)

chain = prompt | model | output_parser
output = chain.invoke({"query": "请列出2023年9月1日-9月10日之间，评分高于8.5的top10电影"})
print(output)