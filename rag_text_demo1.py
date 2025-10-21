import os

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableParallel
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# embeddings_path = "D:\\Learning\\embeddings_db"
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
vectorstore = FAISS.from_texts(["雪华失业了", "熊喜欢吃蜂蜜"], embedding=embeddings)

retriever = vectorstore.as_retriever()

output = retriever.invoke("雪华在哪里上班?")
print(output)

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.7, )

template = """
值根据以下文档回答问题
{context}

问题: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})

chain = setup_and_retrieval | prompt | model | output_parser

output = chain.invoke("雪华在哪里工作")
print(output) # 雪华失业了，所以她没有工作。