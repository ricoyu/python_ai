import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_openai import ChatOpenAI

full_template="""{introduction}

{example}

{start}"""

full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """你在冒充{person}。"""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """
下面是一个交互示例:

Q: {exmaple_q}
A: {example_a}"""

example_prompt = PromptTemplate.from_template(example_template)

start_template = """现在正式开始!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

print(
    pipeline_prompt.format(
        person="Elon Musk",
        exmaple_q="你最喜欢什么车?",
        example_a="Tesla",
        input="你最喜欢的社交媒体网站是什么?")
)

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url)
output_parser = StrOutputParser()
chain = pipeline_prompt | model | output_parser

response = chain.invoke({"input": "你最喜欢的社交媒体网站是什么?", "person": "Elon Musk", "exmaple_q": "你最喜欢什么车?",
                       "example_a": "Tesla", })

print(response)
