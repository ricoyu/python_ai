import os
from operator import itemgetter

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableParallel
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

embeddings_path = "D:\\Learning\\embeddings_db"
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)
# 1. 初始化嵌入模型：使用中文专用模型（避免英文模型对中文语义理解差）
# model_name：模型标识符（Hugging Face Hub 上的公开模型）
# model_kwargs：模型参数（device="cpu" 表示用CPU运行，无GPU可省略）
# encode_kwargs：编码参数（normalize_embeddings=True 使向量归一化，提升检索精度）
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    # 中文轻量模型（1.1GB左右，适合demo）, 默认下载到 C:\Users\<你的用户名>\.cache\huggingface\hub目录
    # model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 把文档转换成嵌入向量
vectorstore = FAISS.from_texts(["雪华失业了", "熊喜欢吃蜂蜜", "雪华9月份还有3天年假", "十月份还有6天年假"],
                               embedding=embeddings)

retriever = vectorstore.as_retriever()

output = retriever.invoke("雪华在哪里上班?")
print(output)  # 雪华失业了，所以她没有工作。

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.7, )

template = """
值根据以下文档回答问题
{context}

问题: {question}
回答必须以称呼"{name}"开头，直接回应问题，不要额外内容。
"""

# 5. 关键修复：定义 context 格式化函数（提取 Document 中的文本）
def format_context(documents):
    return "\n".join([doc.page_content for doc in documents])

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

# 6. 关键修复：RunnableParallel 中明确检索器的输入是 question
setup_and_retrieval = RunnableParallel({
    # 检索器只接收输入中的 "question" 字段作为查询
    "context": itemgetter("question") | retriever | format_context,
    "question": itemgetter("question"),
    "name": itemgetter("name")})

chain = setup_and_retrieval | prompt | model | output_parser

# output = chain.invoke("雪华在哪里工作")
# print(output)
output = chain.invoke({"question": "雪华一共有多少天年假?", "name": "主人"})
print(output)